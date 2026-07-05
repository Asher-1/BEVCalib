#!/usr/bin/env python3
"""A/B compare OpenCV vs C++ EquidistantCamera undistortion for fisheye cameras."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))

from prepare_custom_dataset import ConfigParser
from undistort_utils import (
    build_cpp_undistort_rectify_maps,
    build_opencv_fisheye_undistort_maps,
    get_cpp_focal_scale,
    scale_intrinsics,
    undistort_image,
)


def _space_to_plane_batch(P: np.ndarray, mu, mv, u0, v0, k1, k2, k3, k4) -> np.ndarray:
    """Vectorized EquidistantCamera::spaceToPlane."""
    from undistort_utils import equidistant_r
    norm = np.linalg.norm(P, axis=0)
    theta = np.arccos(np.clip(P[2] / np.maximum(norm, 1e-12), -1.0, 1.0))
    phi = np.arctan2(P[1], P[0])
    r_val = equidistant_r(theta, k1, k2, k3, k4)
    u = mu * r_val * np.cos(phi) + u0
    v = mv * r_val * np.sin(phi) + v0
    return np.stack([u, v], axis=0)


def synthesize_fisheye_test_pattern(w: int, h: int, cam: dict) -> np.ndarray:
    """Paint a colored grid in fisheye space via forward KB projection."""
    intr = cam['intrinsic']
    dist = cam['distortion']
    mu, mv = intr['f_x'], intr['f_y']
    u0, v0 = intr['o_x'], intr['o_y']
    k1, k2, k3, k4 = dist['k1'], dist['k2'], dist['k3'], dist['k4']

    img = np.zeros((h, w, 3), np.uint8)
    thetas = np.linspace(0.05, 1.25, 80)
    phis = np.linspace(-np.pi, np.pi, 160)
    for ti, theta in enumerate(thetas):
        for pi, phi in enumerate(phis):
            P = np.array([[np.sin(theta) * np.cos(phi)],
                          [np.sin(theta) * np.sin(phi)],
                          [np.cos(theta)]])
            uv = _space_to_plane_batch(P, mu, mv, u0, v0, k1, k2, k3, k4)
            u, v = int(uv[0, 0]), int(uv[1, 0])
            if 0 <= u < w and 0 <= v < h:
                img[v, u] = (int(40 + 180 * ti / len(thetas)),
                             int(40 + 180 * (pi % 20) / 20),
                             120)

    # Radial / angular guide lines
    for phi in np.linspace(-np.pi, np.pi, 24, endpoint=False):
        pts = []
        for theta in np.linspace(0.05, 1.25, 200):
            P = np.array([[np.sin(theta) * np.cos(phi)],
                          [np.sin(theta) * np.sin(phi)],
                          [np.cos(theta)]])
            uv = _space_to_plane_batch(P, mu, mv, u0, v0, k1, k2, k3, k4)
            pts.append((int(uv[0, 0]), int(uv[1, 0])))
        for i in range(len(pts) - 1):
            if 0 <= pts[i][0] < w and 0 <= pts[i][1] < h:
                cv2.line(img, pts[i], pts[i + 1], (255, 255, 255), 1, cv2.LINE_AA)
    return img


def load_raw_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def try_extract_from_bag(bag_path: Path, camera_name: str) -> np.ndarray | None:
    try:
        from rosbags.rosbag1 import Reader
        from rosbags.serde import deserialize_cdr
    except ImportError:
        return None
    if bag_path.is_dir():
        bags = sorted(bag_path.glob('*.bag'))
        if not bags:
            return None
        bag_path = bags[0]
    if not bag_path.exists():
        return None
    with Reader(str(bag_path)) as reader:
        for conn, _ts, raw in reader.messages():
            if camera_name in conn.topic and ('compressed' in conn.topic or 'image' in conn.topic):
                msg = deserialize_cdr(raw, conn.msgtype)
                data = bytes(msg.data) if hasattr(msg, 'data') else msg.data
                j = data.find(b'\xff\xd8')
                e = data.rfind(b'\xff\xd9')
                if j >= 0 and e > j:
                    arr = cv2.imdecode(np.frombuffer(data[j:e + 2], np.uint8), cv2.IMREAD_COLOR)
                    if arr is not None:
                        return arr
    return None


def project_lidar_overlay_colored(img: np.ndarray, K: np.ndarray, Tr: np.ndarray, pc_path: Path):
    """Return RGB image with depth-colored lidar overlay; also alignment stats."""
    out = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 and img.shape[2] == 3 else img.copy()
    if img.ndim == 3 and img.shape[2] == 3 and out is not img:
        pass
    elif img.ndim == 3:
        out = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if not pc_path.exists():
        return out, {}
    pts = np.fromfile(str(pc_path), dtype=np.float32).reshape(-1, 4)
    T = np.vstack([Tr.reshape(3, 4), [0, 0, 0, 1]])
    pc = (np.linalg.inv(T) @ np.hstack([pts[:, :3], np.ones((len(pts), 1))]).T).T[:, :3]
    pc = pc[pc[:, 2] > 0]
    P = np.hstack([K, np.zeros((3, 1))])
    uv = (P @ np.hstack([pc, np.ones((len(pc), 1))]).T).T
    uv = uv[:, :2] / uv[:, 2:3]
    depths = pc[:, 2]
    h, w = out.shape[:2]
    m = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    ui, di = uv[m], depths[m]
    fig_ax = None
    return_data = {
        'in_img': int(m.sum()),
        'bottom_third': int((ui[:, 1] > 2 * h / 3).sum()),
        'u_min': float(ui[:, 0].min()) if len(ui) else 0,
        'u_max': float(ui[:, 0].max()) if len(ui) else 0,
    }
    return out, ui, di, return_data


def make_projection_pair_figure(
    img_opencv: np.ndarray,
    img_cpp: np.ndarray,
    K_opencv_out: np.ndarray,
    K_cpp_out: np.ndarray,
    tr_3x4: np.ndarray,
    lidar_bin: Path,
    output_path: Path,
    opencv_is_reference: bool = False,
):
    """Each K must be overlaid on its matching undistorted image."""
    def _to_rgb(img):
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    base_o = _to_rgb(img_opencv)
    base_c = _to_rgb(img_cpp)

    def overlay(base, K):
        pts = np.fromfile(str(lidar_bin), dtype=np.float32).reshape(-1, 4)
        T = np.vstack([tr_3x4.reshape(3, 4), [0, 0, 0, 1]])
        pc = (np.linalg.inv(T) @ np.hstack([pts[:, :3], np.ones((len(pts), 1))]).T).T[:, :3]
        pc = pc[pc[:, 2] > 0]
        P = np.hstack([K, np.zeros((3, 1))])
        uv = (P @ np.hstack([pc, np.ones((len(pc), 1))]).T).T
        uv = uv[:, :2] / uv[:, 2:3]
        depths = pc[:, 2]
        h, w = base.shape[:2]
        m = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        return base, uv[m], depths[m], int(m.sum()), int((uv[m][:, 1] > 2 * h / 3).sum())

    fig, axes = plt.subplots(1, 2, figsize=(20, 7))
    for ax, base, K, name, img_src in [
        (axes[0], base_o, K_opencv_out, 'OpenCV', 'OpenCV undistort + OpenCV K'),
        (axes[1], base_c, K_cpp_out, f'C++ (focal_scale={get_cpp_focal_scale("camera_1")})',
         'C++ undistort + C++ K'),
    ]:
        canvas, uv, d, nin, nb = overlay(base, K)
        ax.imshow(canvas)
        sc = ax.scatter(uv[:, 0], uv[:, 1], c=d, cmap='jet', s=0.3, alpha=0.55)
        ax.set_title(f'{img_src}\nfx={K[0,0]:.0f} cx={K[0,2]:.0f}  in={nin} bottom={nb}')
        ax.axis('off')
    note = ('左：pipeline 真实图（仅 OpenCV 有效）' if opencv_is_reference else
            '两图均从同一 raw 鱼眼图去畸变，K 与图像一一对应')
    fig.suptitle(f'Projection aligned pairs ({note})', fontsize=13)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()


def project_lidar_overlay(img: np.ndarray, K: np.ndarray, Tr: np.ndarray, pc_path: Path) -> np.ndarray:
    out = img.copy()
    if not pc_path.exists():
        return out
    pts = np.fromfile(str(pc_path), dtype=np.float32).reshape(-1, 4)
    T = np.vstack([Tr.reshape(3, 4), [0, 0, 0, 1]])
    pc = (np.linalg.inv(T) @ np.hstack([pts[:, :3], np.ones((len(pts), 1))]).T).T[:, :3]
    pc = pc[pc[:, 2] > 0]
    P = np.hstack([K, np.zeros((3, 1))])
    uv = (P @ np.hstack([pc, np.ones((len(pc), 1))]).T).T
    uv = uv[:, :2] / uv[:, 2:3]
    h, w = img.shape[:2]
    m = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    ui = uv[m].astype(int)
    for x, y in ui[::3]:
        cv2.circle(out, (x, y), 1, (0, 255, 255), -1)
    return out


def run_compare(
    config_dir: Path,
    camera_name: str,
    output_path: Path,
    output_size: tuple[int, int] = (1920, 1080),
    raw_image: Path | None = None,
    bag_path: Path | None = None,
    lidar_bin: Path | None = None,
    tr_3x4: np.ndarray | None = None,
):
    cameras = ConfigParser.parse_cameras_cfg(str(config_dir / 'cameras.cfg'))
    if camera_name not in cameras:
        raise KeyError(f'{camera_name} not in cameras.cfg')
    cam = cameras[camera_name]
    intr = cam['intrinsic']
    dist = cam['distortion']
    w, h = int(intr['img_width']), int(intr['img_height'])
    K = np.array([[intr['f_x'], 0, intr['o_x']],
                  [0, intr['f_y'], intr['o_y']],
                  [0, 0, 1]], dtype=np.float64)
    D = np.array([dist['k1'], dist['k2'], dist['k3'], dist['k4']], dtype=np.float64)
    focal_scale = get_cpp_focal_scale(camera_name)

    raw = None
    if raw_image is not None:
        raw = load_raw_image(str(raw_image))
    elif bag_path is not None:
        raw = try_extract_from_bag(bag_path, camera_name)
    if raw is None:
        print('  未提供 raw/bag，使用合成鱼眼测试图')
        raw = synthesize_fisheye_test_pattern(w, h, cam)
    else:
        print(f'  原始鱼眼图: {raw.shape[1]}x{raw.shape[0]}')

    map_o1, map_o2, K_opencv = build_opencv_fisheye_undistort_maps(K, D, w, h)
    map_c1, map_c2, K_cpp = build_cpp_undistort_rectify_maps(
        K[0, 0], K[1, 1], K[0, 2], K[1, 2],
        D[0], D[1], D[2], D[3], w, h, focal_scale)

    img_opencv = undistort_image(raw, map_o1, map_o2, output_size)
    img_cpp = undistort_image(raw, map_c1, map_c2, output_size)

    ow, oh = output_size
    K_opencv_out = scale_intrinsics(K_opencv, ow, oh, w, h)
    K_cpp_out = scale_intrinsics(K_cpp, ow, oh, w, h)

    print(f'  OpenCV K @1080p: fx={K_opencv_out[0,0]:.1f} cx={K_opencv_out[0,2]:.1f}')
    print(f'  C++    K @1080p: fx={K_cpp_out[0,0]:.1f} cx={K_cpp_out[0,2]:.1f} (focal_scale={focal_scale})')

    if lidar_bin is not None and tr_3x4 is not None:
        img_opencv = project_lidar_overlay(img_opencv, K_opencv_out, tr_3x4, lidar_bin)
        img_cpp = project_lidar_overlay(img_cpp, K_cpp_out, tr_3x4, lidar_bin)
        proj_note = '+ LiDAR overlay (each uses its own K)'
    else:
        proj_note = 'no LiDAR overlay'

    diff = np.abs(img_opencv.astype(np.float32) - img_cpp.astype(np.float32))
    print(f'  图像像素差: mean={diff.mean():.2f} max={diff.max():.0f}')

    fig, axes = plt.subplots(2, 2, figsize=(20, 11))
    axes[0, 0].imshow(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title(f'Input fisheye ({w}x{h})')
    axes[0, 1].imshow(cv2.cvtColor(img_opencv, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title(f'OpenCV balance=0 ({ow}x{oh})\nfx={K_opencv_out[0,0]:.1f} cx={K_opencv_out[0,2]:.1f}')
    axes[1, 0].imshow(cv2.cvtColor(img_cpp, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title(f'C++ Equidistant focal_scale={focal_scale} ({ow}x{oh})\n'
                          f'fx={K_cpp_out[0,0]:.1f} cx={K_cpp_out[0,2]:.1f}')
    axes[1, 1].imshow(diff.mean(axis=2), cmap='hot')
    axes[1, 1].set_title(f'|OpenCV - C++| mean={diff.mean():.2f} {proj_note}')
    for ax in axes.ravel():
        ax.axis('off')
    fig.suptitle(f'Undistort A/B: {camera_name}', fontsize=14)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'  对比图已保存: {output_path}')


def main():
    parser = argparse.ArgumentParser(description='Compare opencv vs cpp undistortion')
    parser.add_argument('--config_dir', required=True)
    parser.add_argument('--camera_name', default='camera_1')
    parser.add_argument('--output', required=True)
    parser.add_argument('--output_width', type=int, default=1920)
    parser.add_argument('--output_height', type=int, default=1080)
    parser.add_argument('--raw_image', default=None, help='原始鱼眼图（可选）')
    parser.add_argument('--bag', default=None, help='bag 文件或目录（可选）')
    parser.add_argument('--lidar_bin', default=None, help='点云 bin 用于投影叠加（可选）')
    parser.add_argument('--tr', default=None, help='Tr 12 floats 或 calib.txt 路径')
    parser.add_argument('--opencv_reference_image', default=None,
                       help='已有 OpenCV 去畸变图（仅用于左图对照，不可叠加 C++ K）')
    parser.add_argument('--projection_pair_output', default=None,
                       help='生成 K+图像配对投影图（每种 K 只用各自去畸变图）')
    args = parser.parse_args()

    tr = None
    if args.tr:
        p = Path(args.tr)
        if p.exists():
            for line in open(p):
                if line.startswith('Tr:'):
                    tr = np.array([float(x) for x in line.split()[1:]])
                    break
        else:
            tr = np.array([float(x) for x in args.tr.split()])

    run_compare(
        Path(args.config_dir),
        args.camera_name,
        Path(args.output),
        (args.output_width, args.output_height),
        Path(args.raw_image) if args.raw_image else None,
        Path(args.bag) if args.bag else None,
        Path(args.lidar_bin) if args.lidar_bin else None,
        tr,
    )

    if args.projection_pair_output and args.lidar_bin and tr is not None:
        cameras = ConfigParser.parse_cameras_cfg(str(Path(args.config_dir) / 'cameras.cfg'))
        cam = cameras[args.camera_name]
        intr, dist = cam['intrinsic'], cam['distortion']
        w, h = int(intr['img_width']), int(intr['img_height'])
        K = np.array([[intr['f_x'], 0, intr['o_x']], [0, intr['f_y'], intr['o_y']], [0, 0, 1]], float)
        D = np.array([dist['k1'], dist['k2'], dist['k3'], dist['k4']])
        raw = None
        if args.raw_image:
            raw = load_raw_image(args.raw_image)
        elif args.bag:
            raw = try_extract_from_bag(Path(args.bag), args.camera_name)
        if raw is None:
            raw = synthesize_fisheye_test_pattern(w, h, cam)
        map_o1, map_o2, K_o = build_opencv_fisheye_undistort_maps(K, D, w, h)
        map_c1, map_c2, K_c = build_cpp_undistort_rectify_maps(
            K[0, 0], K[1, 1], K[0, 2], K[1, 2], D[0], D[1], D[2], D[3], w, h,
            get_cpp_focal_scale(args.camera_name))
        ow, oh = args.output_width, args.output_height
        img_o = undistort_image(raw, map_o1, map_o2, (ow, oh))
        img_c = undistort_image(raw, map_c1, map_c2, (ow, oh))
        if args.opencv_reference_image:
            img_o = load_raw_image(args.opencv_reference_image)
        make_projection_pair_figure(
            img_o, img_c,
            scale_intrinsics(K_o, ow, oh, w, h),
            scale_intrinsics(K_c, ow, oh, w, h),
            tr, Path(args.lidar_bin), Path(args.projection_pair_output),
            opencv_is_reference=bool(args.opencv_reference_image),
        )
        print(f'  配对投影图: {args.projection_pair_output}')


if __name__ == '__main__':
    main()
