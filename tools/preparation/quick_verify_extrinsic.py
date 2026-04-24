#!/usr/bin/env python3
"""
快速验证 LiDAR 外参: 点云投影到图像的匹配度检测

在运行 prepare_custom_dataset.py 之前，先用此脚本验证外参是否正确，
避免因外参问题导致生成的训练数据无效，浪费大量时间。

功能:
  1. 解析 configs/ 和 model/ 两套外参 (在线标定 vs 出厂标定)
  2. 从 bag 中提取样本帧 (图像+点云)
  3. 分别用两套外参投影，生成对比可视化
  4. 网格搜索角度补偿，找到最佳匹配
  5. 输出补偿建议 (patch 到 lidars.cfg)

用法:
    python quick_verify_extrinsic.py /path/to/trip_dir
    python quick_verify_extrinsic.py /path/to/trip_dir --camera traffic_2
    python quick_verify_extrinsic.py /path/to/trip_dir --compensate --grid 1.0
    python quick_verify_extrinsic.py /path/to/trip_dir --apply-pitch 0.3 --apply-roll -0.1
"""

import os
import sys
import re
import argparse
import struct
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy.spatial.transform import Rotation as R
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.font_manager as fm
import time

# CJK 字体支持
_CJK_FONT = None
for _fp in ['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
            '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
            '/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc']:
    if os.path.exists(_fp):
        _CJK_FONT = _fp
        break
if _CJK_FONT:
    fm.fontManager.addfont(_CJK_FONT)
    plt.rcParams['font.family'] = fm.FontProperties(fname=_CJK_FONT).get_name()
plt.rcParams['axes.unicode_minus'] = False


def parse_cameras_cfg(filepath: str) -> Dict:
    """解析 cameras.cfg (同时包含外参和内参)"""
    with open(filepath, 'r') as f:
        content = f.read()

    cameras = {}
    blocks = re.findall(r'config\s*\{(.*?)\n\}', content, re.DOTALL)

    for block in blocks:
        cam = _parse_camera_block(block)
        if cam:
            cameras[cam['camera_dev']] = cam
    return cameras


def _find_nested_block(text: str, name: str) -> Optional[str]:
    """找到 text 中名为 name 的 { ... } 块（支持嵌套）"""
    m = re.search(rf'{name}\s*\{{', text)
    if not m:
        return None
    start = m.end()
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
        i += 1
    return text[start:i - 1] if depth == 0 else None


def _parse_camera_block(block: str) -> Optional[Dict]:
    def get_val(pat, text, default=None):
        m = re.search(pat, text)
        return m.group(1).strip().strip('"') if m else default

    def get_f(pat, text, default=0.0):
        v = get_val(pat, text)
        return float(v) if v is not None else default

    camera_dev = get_val(r'camera_dev:\s*"([^"]+)"', block)
    if not camera_dev:
        return None

    # 外参 (sensor_to_cam → Camera→Sensing)
    ext_block = _find_nested_block(block, 'sensor_to_cam') or ''
    pos_block = _find_nested_block(ext_block, 'position') if ext_block else None
    ori_block = _find_nested_block(ext_block, 'orientation') if ext_block else None

    if pos_block:
        pos_x = get_f(r'x:\s*([-\d.e]+)', pos_block)
        pos_y = get_f(r'y:\s*([-\d.e]+)', pos_block)
        pos_z = get_f(r'z:\s*([-\d.e]+)', pos_block)
    else:
        pos_x = get_f(r'position\s*\{[^}]*x:\s*([-\d.e]+)', block)
        pos_y = get_f(r'position\s*\{[^}]*y:\s*([-\d.e]+)', block)
        pos_z = get_f(r'position\s*\{[^}]*z:\s*([-\d.e]+)', block)

    if ori_block:
        ori_qx = get_f(r'qx:\s*([-\d.e]+)', ori_block)
        ori_qy = get_f(r'qy:\s*([-\d.e]+)', ori_block)
        ori_qz = get_f(r'qz:\s*([-\d.e]+)', ori_block)
        ori_qw = get_f(r'qw:\s*([-\d.e]+)', ori_block, 1.0)
    else:
        ori_qx = get_f(r'orientation\s*\{[^}]*qx:\s*([-\d.e]+)', block)
        ori_qy = get_f(r'orientation\s*\{[^}]*qy:\s*([-\d.e]+)', block)
        ori_qz = get_f(r'orientation\s*\{[^}]*qz:\s*([-\d.e]+)', block)
        ori_qw = get_f(r'orientation\s*\{[^}]*qw:\s*([-\d.e]+)', block, 1.0)

    # 内参
    fx = get_f(r'f_x:\s*([-\d.e]+)', block)
    fy = get_f(r'f_y:\s*([-\d.e]+)', block)
    ox = get_f(r'o_x:\s*([-\d.e]+)', block)
    oy = get_f(r'o_y:\s*([-\d.e]+)', block)
    w = int(get_f(r'img_width:\s*(\d+)', block, 1920))
    h = int(get_f(r'img_height:\s*(\d+)', block, 1080))

    model_type = get_val(r'model_type:\s*(\w+)', block, 'PINHOLE')

    k1 = get_f(r'k_1:\s*([-\d.e]+)', block, 0.0)
    k2 = get_f(r'k_2:\s*([-\d.e]+)', block, 0.0)
    p1 = get_f(r'p_1:\s*([-\d.e]+)', block, 0.0)
    p2 = get_f(r'p_2:\s*([-\d.e]+)', block, 0.0)
    k3 = get_f(r'k_3:\s*([-\d.e]+)', block, 0.0)
    k4 = get_f(r'k_4:\s*([-\d.e]+)', block, 0.0)

    return {
        'camera_dev': camera_dev,
        'position': np.array([pos_x, pos_y, pos_z]),
        'orientation': np.array([ori_qx, ori_qy, ori_qz, ori_qw]),
        'intrinsic': {'f_x': fx, 'f_y': fy, 'o_x': ox, 'o_y': oy,
                      'img_width': w, 'img_height': h},
        'distortion': {'k1': k1, 'k2': k2, 'p1': p1, 'p2': p2, 'k3': k3, 'k4': k4,
                       'model_type': 'fisheye' if model_type == 'KANNALA_BRANDT' else 'pinhole'},
    }


def parse_lidars_cfg(filepath: str) -> Dict:
    """解析 lidars.cfg，提取 vehicle_to_sensing 和 sensor_to_lidar"""
    with open(filepath, 'r') as f:
        content = f.read()

    def get_f(pat, text, default=0.0):
        m = re.search(pat, text)
        return float(m.group(1)) if m else default

    result = {}

    # vehicle_to_sensing (Sensing→Vehicle)
    v2s = re.search(r'vehicle_to_sensing\s*\{(.*?)\n\}', content, re.DOTALL)
    if v2s:
        blk = v2s.group(1)
        result['sensing_to_vehicle'] = {
            'position': np.array([get_f(r'x:\s*([-\d.e]+)', blk),
                                  get_f(r'y:\s*([-\d.e]+)', blk),
                                  get_f(r'z:\s*([-\d.e]+)', blk)]),
            'orientation': np.array([get_f(r'qx:\s*([-\d.e]+)', blk),
                                     get_f(r'qy:\s*([-\d.e]+)', blk),
                                     get_f(r'qz:\s*([-\d.e]+)', blk),
                                     get_f(r'qw:\s*([-\d.e]+)', blk, 1.0)])
        }

    # sensor_to_lidar (LiDAR→Sensing) — 用嵌套块解析
    config_block = _find_nested_block(content, 'config')
    if config_block:
        s2l_blk = _find_nested_block(config_block, 'sensor_to_lidar')
        if s2l_blk:
            pos_blk = _find_nested_block(s2l_blk, 'position')
            ori_blk = _find_nested_block(s2l_blk, 'orientation')
            if pos_blk:
                result['position'] = np.array([get_f(r'x:\s*([-\d.e]+)', pos_blk),
                                               get_f(r'y:\s*([-\d.e]+)', pos_blk),
                                               get_f(r'z:\s*([-\d.e]+)', pos_blk)])
            if ori_blk:
                result['orientation'] = np.array([get_f(r'qx:\s*([-\d.e]+)', ori_blk),
                                                  get_f(r'qy:\s*([-\d.e]+)', ori_blk),
                                                  get_f(r'qz:\s*([-\d.e]+)', ori_blk),
                                                  get_f(r'qw:\s*([-\d.e]+)', ori_blk, 1.0)])

    # install_angle_error
    iae = re.search(r'install_angle_error\s*\{(.*?)\}', content, re.DOTALL)
    if iae:
        blk = iae.group(1)
        result['install_angle_error'] = np.array([
            get_f(r'x:\s*([-\d.e]+)', blk),
            get_f(r'y:\s*([-\d.e]+)', blk),
            get_f(r'z:\s*([-\d.e]+)', blk)])

    return result


def build_transforms(cam_cfg: Dict, lidar_cfg: Dict):
    """从外参配置构建变换矩阵

    Returns:
        T_lidar_to_cam: LiDAR→Camera (4x4)
        T_lidar_to_sensing: LiDAR→Sensing (4x4)
        K: 内参矩阵 (3x3)
    """
    T_cam_to_sensing = np.eye(4)
    r = R.from_quat(cam_cfg['orientation'])
    T_cam_to_sensing[:3, :3] = r.as_matrix()
    T_cam_to_sensing[:3, 3] = cam_cfg['position']

    T_sensing_to_cam = np.linalg.inv(T_cam_to_sensing)

    T_lidar_to_sensing = np.eye(4)
    r = R.from_quat(lidar_cfg['orientation'])
    T_lidar_to_sensing[:3, :3] = r.as_matrix()
    T_lidar_to_sensing[:3, 3] = lidar_cfg['position']

    T_lidar_to_cam = T_sensing_to_cam @ T_lidar_to_sensing

    intr = cam_cfg['intrinsic']
    K = np.array([[intr['f_x'], 0, intr['o_x']],
                  [0, intr['f_y'], intr['o_y']],
                  [0, 0, 1]])

    return T_lidar_to_cam, T_lidar_to_sensing, K


def init_undistortion(cam_cfg: Dict, K_orig: np.ndarray):
    """初始化图像去畸变映射，返回 (map1, map2, K_new)"""
    dist = cam_cfg.get('distortion', {})
    model_type = dist.get('model_type', 'pinhole')
    w = cam_cfg['intrinsic']['img_width']
    h = cam_cfg['intrinsic']['img_height']

    if model_type == 'fisheye':
        D = np.array([dist.get('k1', 0), dist.get('k2', 0),
                      dist.get('k3', 0), dist.get('k4', 0)])
    else:
        D = np.array([dist.get('k1', 0), dist.get('k2', 0),
                      dist.get('p1', 0), dist.get('p2', 0), dist.get('k3', 0)])

    if np.allclose(D, 0, atol=1e-10):
        return None, None, K_orig

    if model_type == 'fisheye':
        D_col = D.reshape(4, 1)
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K_orig, D_col, (w, h), np.eye(3), balance=0, new_size=(w, h))
        m1, m2 = cv2.fisheye.initUndistortRectifyMap(
            K_orig, D_col, np.eye(3), new_K, (w, h), cv2.CV_16SC2)
    else:
        new_K, _ = cv2.getOptimalNewCameraMatrix(
            K_orig, D, (w, h), alpha=0, newImgSize=(w, h))
        m1, m2 = cv2.initUndistortRectifyMap(
            K_orig, D, None, new_K, (w, h), cv2.CV_16SC2)

    return m1, m2, new_K


def apply_angle_compensation(T_lidar_to_sensing: np.ndarray,
                             pitch_deg: float, roll_deg: float, yaw_deg: float) -> np.ndarray:
    """在 LiDAR→Sensing 上叠加角度补偿

    补偿旋转应用在 Sensing 系中:
        T_compensated = R_comp @ T_lidar_to_sensing
    """
    if abs(pitch_deg) < 1e-6 and abs(roll_deg) < 1e-6 and abs(yaw_deg) < 1e-6:
        return T_lidar_to_sensing.copy()

    R_comp = R.from_euler('xyz', [roll_deg, pitch_deg, yaw_deg], degrees=True).as_matrix()
    T_comp = np.eye(4)
    T_comp[:3, :3] = R_comp

    return T_comp @ T_lidar_to_sensing


def project_points(points: np.ndarray, T_lidar_to_cam: np.ndarray,
                   K: np.ndarray, img_shape: Tuple[int, int]):
    """投影点云到图像平面

    Returns:
        pts_2d: (M, 2) 图像坐标
        depths: (M,) 深度
        n_total: 总点数
    """
    H, W = img_shape
    pts_hom = np.hstack([points[:, :3], np.ones((len(points), 1))])
    pts_cam = (T_lidar_to_cam @ pts_hom.T).T[:, :3]

    mask = pts_cam[:, 2] > 0.5
    pts_cam = pts_cam[mask]
    if len(pts_cam) == 0:
        return np.zeros((0, 2)), np.zeros(0), len(points)

    pts_img = (K @ pts_cam.T).T
    pts_2d = pts_img[:, :2] / pts_img[:, 2:3]
    depths = pts_cam[:, 2]

    mask2 = (pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < W) & \
            (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < H)
    return pts_2d[mask2], depths[mask2], len(points)


def extract_sample_frames(trip_dir: Path, camera_name: str, n_frames: int = 3,
                          lidar_topic: str = '/sensors/lidar/combined_point_cloud_proto'):
    """从 bag 中提取样本帧

    图像通常在 Heavy_Topic_Group bags, 点云在 Medium_Topic_Group bags.
    需要按时间戳配对同名的 Heavy+Medium bag.
    """
    try:
        from rosbags.rosbag1 import Reader
    except ImportError:
        print("需要安装 rosbags: pip install rosbags")
        sys.exit(1)

    bag_dir = trip_dir / 'bags'
    bag_files = sorted(bag_dir.glob('**/*.bag'))
    if not bag_files:
        print(f"未找到 bag 文件: {bag_dir}")
        sys.exit(1)

    possible_img_topics = [
        f'/sensors/camera/{camera_name}_raw_data/compressed_proto',
        f'/sensors/camera/{camera_name}/compressed_proto',
    ]

    # 按时间戳分组 Heavy+Medium bags
    heavy = {_bag_timestamp(f): f for f in bag_files if 'Heavy' in f.name}
    medium = {_bag_timestamp(f): f for f in bag_files if 'Medium' in f.name}

    if not heavy or not medium:
        print(f"  Heavy bags: {len(heavy)}, Medium bags: {len(medium)}")
        print(f"  尝试从所有 bags 中混合提取...")
        return _extract_mixed(bag_files, possible_img_topics, lidar_topic, n_frames)

    common_ts = sorted(set(heavy.keys()) & set(medium.keys()))
    if not common_ts:
        print(f"  Heavy 和 Medium bags 无时间匹配，尝试混合提取...")
        return _extract_mixed(bag_files, possible_img_topics, lidar_topic, n_frames)

    indices = np.linspace(0, len(common_ts) - 1, min(n_frames, len(common_ts)), dtype=int)

    frames = []
    for idx in indices:
        ts = common_ts[idx]
        h_bag, m_bag = heavy[ts], medium[ts]
        frame = _extract_from_pair(h_bag, m_bag, possible_img_topics, lidar_topic)
        if frame is not None:
            frames.append(frame)
            print(f"  [Frame {len(frames)}] 从 {h_bag.name} + {m_bag.name} 提取成功 "
                  f"(图像 {frame[0].shape[:2]}, 点云 {len(frame[1])} 点)")
        if len(frames) >= n_frames:
            break

    return frames


def _bag_timestamp(bag_path: Path) -> str:
    """从 bag 文件名提取时间戳部分 (如 20260406_081015)"""
    name = bag_path.stem
    m = re.search(r'(\d{8}_\d{6})', name)
    return m.group(1) if m else name


def _extract_from_pair(heavy_bag: Path, medium_bag: Path,
                       img_topics: List[str], lidar_topic: str,
                       max_time_diff: float = 0.055):
    """从 Heavy bag 提取图像, 从 Medium bag 提取点云, 按时间戳对齐

    Args:
        max_time_diff: 图像-点云最大时间差 (秒), 默认55ms (对齐C++ kMaxLidarCameraDelta)
    """
    from rosbags.rosbag1 import Reader

    # 1) 从 Medium bag 先提取几帧点云及其时间戳
    pc_list = []  # [(ts_sec, points), ...]
    try:
        with Reader(medium_bag) as reader:
            available = {c.topic for c in reader.connections}
            if lidar_topic in available:
                for conn, ts, rawdata in reader.messages():
                    if conn.topic == lidar_topic:
                        ts_sec = ts / 1e9
                        data = _extract_string_msg(rawdata)
                        if data:
                            pc = _parse_pointcloud_simple(data)
                            if pc is not None:
                                pc_list.append((ts_sec, pc))
                                if len(pc_list) >= 10:
                                    break
    except Exception as e:
        print(f"  Medium bag 读取失败 ({medium_bag.name}): {e}")

    if not pc_list:
        return None

    # 2) 从 Heavy bag 提取图像, 找到与某帧点云时间最接近的
    best_pair = None
    best_dt = float('inf')

    try:
        with Reader(heavy_bag) as reader:
            available = {c.topic for c in reader.connections}
            active_topic = next((t for t in img_topics if t in available), None)
            if active_topic:
                img_count = 0
                for conn, ts, rawdata in reader.messages():
                    if conn.topic == active_topic:
                        img_ts = ts / 1e9
                        data = _extract_string_msg(rawdata)
                        if data:
                            # 找最近的点云帧
                            for pc_ts, pc in pc_list:
                                dt = abs(img_ts - pc_ts)
                                if dt < best_dt and dt < max_time_diff:
                                    img = _decode_image_from_bytes(data)
                                    if img is not None:
                                        best_dt = dt
                                        best_pair = (img, pc, img_ts, pc_ts)
                        img_count += 1
                        if best_pair is not None or img_count > 20:
                            break
    except Exception as e:
        print(f"  Heavy bag 读取失败 ({heavy_bag.name}): {e}")

    if best_pair is not None:
        img, pc, img_ts, pc_ts = best_pair
        print(f"    time-sync: dt={best_dt*1000:.1f}ms (img={img_ts:.3f}, pc={pc_ts:.3f})")
        return (img, pc)
    else:
        # fallback: 取第一帧点云和第一帧图像 (不保证时间对齐)
        print(f"    warn: no time-aligned pair (threshold={max_time_diff*1000:.0f}ms), using first frames")
        try:
            with Reader(heavy_bag) as reader:
                active_topic = next((t for t in img_topics if t in {c.topic for c in reader.connections}), None)
                if active_topic:
                    for conn, ts, rawdata in reader.messages():
                        if conn.topic == active_topic:
                            data = _extract_string_msg(rawdata)
                            if data:
                                img = _decode_image_from_bytes(data)
                                if img is not None:
                                    return (img, pc_list[0][1])
        except:
            pass
    return None


def _extract_mixed(bag_files: List[Path], img_topics: List[str],
                   lidar_topic: str, n_frames: int):
    """fallback: 从任意 bags 中分别提取图像和点云"""
    frames = []
    for bag in bag_files[:20]:
        frame = _extract_one_frame(bag, img_topics, lidar_topic)
        if frame is not None:
            frames.append(frame)
            print(f"  [Frame {len(frames)}] 从 {bag.name} 提取成功")
        if len(frames) >= n_frames:
            break
    return frames


def _extract_string_msg(rawdata: bytes) -> Optional[bytes]:
    if len(rawdata) < 4:
        return None
    length = struct.unpack('<I', rawdata[:4])[0]
    if 4 + length <= len(rawdata):
        return rawdata[4:4 + length]
    return rawdata


def _decode_image_from_bytes(data: bytes) -> Optional[np.ndarray]:
    """从 bytes 中找 JPEG 并解码"""
    jpeg_start = data.find(b'\xff\xd8')
    if jpeg_start == -1:
        return None
    jpeg_end = data.rfind(b'\xff\xd9')
    if jpeg_end <= jpeg_start:
        return None
    img_array = np.frombuffer(data[jpeg_start:jpeg_end + 2], np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if image is not None:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return None


def _parse_pointcloud_simple(data: bytes) -> Optional[np.ndarray]:
    """简化的点云解析: 尝试从 proto 数据中提取 XYZ 坐标

    策略: 尝试在数据中找到大块连续的 float32 点云数据 (16字节/点)
    """
    candidates = [data, data[4:] if len(data) > 4 else b'',
                  data[8:] if len(data) > 8 else b'']

    for cand in candidates:
        result = _try_parse_wire_format(cand)
        if result is not None and len(result) >= 100:
            return result

    for offset in [0, 4, 8, 20, 40, 60, 80, 100, 200, 300, 500]:
        if len(data) < offset + 64:
            continue
        rem = len(data) - offset
        if rem % 16 != 0:
            continue
        n = rem // 16
        if n < 100:
            continue

        pts = []
        ok = 0
        for i in range(min(500, n)):
            o = offset + i * 16
            x, y, z, _ = struct.unpack_from('ffff', data, o)
            if not (np.isnan(x) or np.isnan(y) or np.isnan(z)) and \
               abs(x) < 500 and abs(y) < 500 and abs(z) < 100:
                ok += 1

        if ok > 50:
            pts_all = np.frombuffer(data[offset:offset + n * 16],
                                    dtype=np.float32).reshape(-1, 4)
            mask = ~np.isnan(pts_all).any(axis=1)
            mask &= (np.abs(pts_all[:, 0]) < 500) & \
                    (np.abs(pts_all[:, 1]) < 500) & \
                    (np.abs(pts_all[:, 2]) < 100)
            pts_valid = pts_all[mask]
            if len(pts_valid) >= 100:
                return pts_valid

    return None


def _decode_varint(buf, pos):
    n, sh = 0, 0
    while pos < len(buf):
        b = buf[pos]; pos += 1
        n |= (b & 0x7F) << sh
        if not (b & 0x80):
            return n, pos
        sh += 7
        if sh >= 64:
            break
    return n, pos


def _try_parse_wire_format(data: bytes) -> Optional[np.ndarray]:
    """尝试 protobuf wire format 解析点云"""
    pos = 0
    point_step = 0
    raw_data = None
    fields = []

    while pos < len(data):
        if pos + 1 > len(data):
            break
        tag, pos = _decode_varint(data, pos)
        field_num, wire = tag >> 3, tag & 7

        if wire == 0:  # varint
            val, pos = _decode_varint(data, pos)
            if field_num == 4:  # point_step
                point_step = val
        elif wire == 1:  # fixed64
            pos += 8
        elif wire == 5:  # fixed32
            pos += 4
        elif wire == 2:  # length-delimited
            L, pos = _decode_varint(data, pos)
            if pos + L > len(data):
                break
            chunk = data[pos:pos + L]
            if field_num == 3:  # fields (PointField)
                name, offset_f, dt = _parse_point_field(chunk)
                if name:
                    fields.append((name, offset_f, dt))
            elif field_num == 5:  # data
                raw_data = chunk
            pos += L
        else:
            break

    if raw_data is None or point_step <= 0:
        return None

    n = len(raw_data) // point_step
    if n < 50:
        return None

    fields_map = {}
    dt_fmt = {1: ('b', 1), 2: ('B', 1), 3: ('h', 1), 4: ('H', 1),
              5: ('i', 1), 6: ('I', 1), 7: ('f', 1), 8: ('d', 1)}
    for name, off, dt in fields:
        if dt in dt_fmt:
            fmt, scale = dt_fmt[dt]
            if dt in (3, 4):
                scale = 0.01
            fields_map[name] = (off, fmt, scale)

    if 'x' not in fields_map:
        if point_step == 16:
            fields_map = {'x': (0, 'f', 1), 'y': (4, 'f', 1), 'z': (8, 'f', 1)}
        elif point_step == 10:
            fields_map = {'x': (0, 'h', 0.01), 'y': (2, 'h', 0.01), 'z': (4, 'h', 0.01)}
        else:
            return None

    x_off, x_fmt, x_sc = fields_map.get('x', (0, 'f', 1))
    y_off, y_fmt, y_sc = fields_map.get('y', (4, 'f', 1))
    z_off, z_fmt, z_sc = fields_map.get('z', (8, 'f', 1))

    raw_arr = np.frombuffer(raw_data, dtype=np.uint8).reshape(n, point_step)

    def read_field(arr, off, fmt):
        np_dt = {'f': np.float32, 'h': np.int16, 'H': np.uint16,
                 'i': np.int32, 'I': np.uint32, 'b': np.int8, 'B': np.uint8, 'd': np.float64}
        sz = {'f': 4, 'h': 2, 'H': 2, 'i': 4, 'I': 4, 'b': 1, 'B': 1, 'd': 8}
        return arr[:, off:off + sz[fmt]].copy().view(np_dt[fmt]).flatten()

    xs = read_field(raw_arr, x_off, x_fmt).astype(np.float32) * x_sc
    ys = read_field(raw_arr, y_off, y_fmt).astype(np.float32) * y_sc
    zs = read_field(raw_arr, z_off, z_fmt).astype(np.float32) * z_sc

    pts = np.column_stack([xs, ys, zs, np.zeros(n, dtype=np.float32)])
    mask = ~np.isnan(pts[:, :3]).any(axis=1)
    mask &= (np.abs(xs) < 500) & (np.abs(ys) < 500) & (np.abs(zs) < 100)
    return pts[mask]


def _parse_point_field(data: bytes):
    """解析 PointField 消息"""
    name, offset_f, datatype = '', 0, 7
    pos = 0
    while pos < len(data):
        tag, pos = _decode_varint(data, pos)
        fn, wire = tag >> 3, tag & 7
        if wire == 0:
            val, pos = _decode_varint(data, pos)
            if fn == 2:
                offset_f = val
            elif fn == 3:
                datatype = val
        elif wire == 2:
            L, pos = _decode_varint(data, pos)
            if fn == 1:
                name = data[pos:pos + L].decode('utf-8', errors='ignore')
            pos += L
        elif wire == 5:
            pos += 4
        elif wire == 1:
            pos += 8
        else:
            break
    return name, offset_f, datatype


def _extract_one_frame(bag_path: Path, img_topics: List[str],
                       lidar_topic: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """从单个 bag 中提取一对 (image, pointcloud)"""
    from rosbags.rosbag1 import Reader

    image = None
    pointcloud = None

    try:
        with Reader(bag_path) as reader:
            available = {c.topic for c in reader.connections}
            active_img_topic = None
            for t in img_topics:
                if t in available:
                    active_img_topic = t
                    break

            if active_img_topic is None or lidar_topic not in available:
                return None

            img_count, pc_count = 0, 0
            for connection, timestamp, rawdata in reader.messages():
                if connection.topic == active_img_topic and image is None:
                    data = _extract_string_msg(rawdata)
                    if data:
                        img = _decode_image_from_bytes(data)
                        if img is not None:
                            image = img
                            img_count += 1

                elif connection.topic == lidar_topic and pointcloud is None:
                    data = _extract_string_msg(rawdata)
                    if data:
                        pc = _parse_pointcloud_simple(data)
                        if pc is not None:
                            pointcloud = pc
                            pc_count += 1

                if image is not None and pointcloud is not None:
                    return (image, pointcloud)

                if img_count > 5 or pc_count > 5:
                    break
    except Exception as e:
        print(f"  bag 读取失败 ({bag_path.name}): {e}")

    if image is not None and pointcloud is not None:
        return (image, pointcloud)
    return None


def render_projection(ax, image: np.ndarray, pts_2d: np.ndarray,
                      depths: np.ndarray, title: str, n_total: int,
                      show_edge_overlay: bool = False):
    """在 matplotlib axes 上绘制投影结果"""
    display_img = image.copy()

    if show_edge_overlay:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        edges = cv2.Canny(gray, 50, 150)
        if display_img.ndim == 3:
            display_img[edges > 0] = [0, 255, 0]

    ax.imshow(display_img)
    if len(pts_2d) > 0:
        score = compute_edge_alignment_score(image, pts_2d, depths)
        scatter = ax.scatter(pts_2d[:, 0], pts_2d[:, 1],
                            c=depths, cmap='jet', s=0.5, alpha=0.5,
                            vmin=np.percentile(depths, 5),
                            vmax=np.percentile(depths, 95))
        ax.set_title(f'{title}\n{len(pts_2d)}/{n_total} pts | EdgeScore={score:.4f}',
                     fontsize=9)
    else:
        ax.set_title(f'{title}\nNo valid projection', fontsize=9, color='red')
    ax.axis('off')


def compare_extrinsics(configs_cam, configs_lidar, model_cam, model_lidar):
    """比较两套外参的差异"""
    print(f"\n{'='*70}")
    print(f"外参比较: configs (在线标定) vs model (出厂)")
    print(f"{'='*70}")

    # Camera 外参差异
    r_cfg = R.from_quat(configs_cam['orientation'])
    r_mdl = R.from_quat(model_cam['orientation'])
    r_diff = r_cfg * r_mdl.inv()
    euler_diff = r_diff.as_euler('xyz', degrees=True)
    pos_diff = configs_cam['position'] - model_cam['position']

    print(f"\n  Camera (Camera→Sensing) 差异:")
    print(f"    平移差异: [{pos_diff[0]:.6f}, {pos_diff[1]:.6f}, {pos_diff[2]:.6f}] m")
    print(f"    旋转差异 (RPY): [{euler_diff[0]:.4f}°, {euler_diff[1]:.4f}°, {euler_diff[2]:.4f}°]")

    cam_iae = None
    for src in [configs_cam]:
        iae_str = None

    # LiDAR 外参差异
    r_cfg = R.from_quat(configs_lidar['orientation'])
    r_mdl = R.from_quat(model_lidar['orientation'])
    r_diff = r_cfg * r_mdl.inv()
    euler_diff = r_diff.as_euler('xyz', degrees=True)
    pos_diff = configs_lidar['position'] - model_lidar['position']

    print(f"\n  LiDAR (LiDAR→Sensing) 差异:")
    print(f"    平移差异: [{pos_diff[0]:.6f}, {pos_diff[1]:.6f}, {pos_diff[2]:.6f}] m")
    print(f"    旋转差异 (RPY): [{euler_diff[0]:.4f}°, {euler_diff[1]:.4f}°, {euler_diff[2]:.4f}°]")

    if 'install_angle_error' in configs_lidar:
        iae = configs_lidar['install_angle_error']
        print(f"    install_angle_error: [{iae[0]:.6f}°, {iae[1]:.6f}°, {iae[2]:.6f}°]")

    # 综合 T_lidar_to_cam 差异
    T_l2c_cfg, _, _ = build_transforms(configs_cam, configs_lidar)
    T_l2c_mdl, _, _ = build_transforms(model_cam, model_lidar)
    R_l2c_diff = R.from_matrix(T_l2c_cfg[:3, :3]) * R.from_matrix(T_l2c_mdl[:3, :3]).inv()
    l2c_euler = R_l2c_diff.as_euler('xyz', degrees=True)
    l2c_pos = T_l2c_cfg[:3, 3] - T_l2c_mdl[:3, 3]

    print(f"\n  综合 LiDAR→Camera 差异 (影响投影):")
    print(f"    平移差异: [{l2c_pos[0]:.6f}, {l2c_pos[1]:.6f}, {l2c_pos[2]:.6f}] m")
    print(f"    旋转差异 (RPY): [{l2c_euler[0]:.4f}°, {l2c_euler[1]:.4f}°, {l2c_euler[2]:.4f}°]")
    print(f"    旋转差异总量: {np.linalg.norm(l2c_euler):.4f}°")

    quality = "GOOD" if np.linalg.norm(l2c_euler) < 1.0 else "NEEDS_CHECK"
    print(f"\n  评估: {quality} (旋转差异 {'< 1°' if quality == 'GOOD' else '>= 1°, 建议检查'})")
    return l2c_euler


def compute_edge_alignment_score(image: np.ndarray, pts_2d: np.ndarray,
                                  depths: np.ndarray) -> float:
    """基于图像梯度的边缘对齐评分

    原理: 正确外参 → 点云投影落在物体边缘/纹理处 → 图像梯度值高
    评分 = 投影点处的平均梯度强度 (越高越好)

    补充指标: 深度不连续处 (遮挡边界) 的点应落在图像边缘
    """
    if len(pts_2d) < 10:
        return 0.0

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    H, W = gray.shape

    # Sobel 梯度幅值 (归一化到 [0, 1])
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    grad_mag = np.sqrt(gx**2 + gy**2)
    grad_max = grad_mag.max()
    if grad_max > 0:
        grad_mag /= grad_max

    # 在投影点位置采样梯度值
    xs = np.clip(pts_2d[:, 0].astype(int), 0, W - 1)
    ys = np.clip(pts_2d[:, 1].astype(int), 0, H - 1)
    grad_at_pts = grad_mag[ys, xs]

    # 综合评分: 平均梯度 × 投影有效率
    avg_grad = np.mean(grad_at_pts)
    coverage = len(pts_2d) / max(1, len(pts_2d))

    # 加分: 对深度不连续处 (遮挡边界) 的点单独评分
    # 这些点更应该落在图像边缘上
    if len(depths) > 50:
        depth_sorted_idx = np.argsort(depths)
        depth_gradients = np.abs(np.diff(depths[depth_sorted_idx]))
        threshold = np.percentile(depth_gradients, 90)
        boundary_mask = np.zeros(len(depths), dtype=bool)
        boundary_idx = depth_sorted_idx[:-1][depth_gradients > threshold]
        boundary_mask[boundary_idx] = True

        if boundary_mask.sum() > 5:
            boundary_grad = grad_at_pts[boundary_mask].mean()
            score = 0.6 * avg_grad + 0.4 * boundary_grad
        else:
            score = avg_grad
    else:
        score = avg_grad

    return score


def grid_search_compensation(frame_data: list, T_sensing_to_cam, T_lidar_to_sensing,
                             K, grid_step=0.5, grid_range=1.5):
    """网格搜索最佳角度补偿 (多帧联合边缘对齐评分)

    Args:
        frame_data: [(image, pointcloud, img_shape), ...] 预处理好的多帧数据
    """
    angles = np.arange(-grid_range, grid_range + grid_step * 0.5, grid_step)
    best_score = -1
    best_comp = (0, 0, 0)
    results = {}

    def _eval(pitch, roll, yaw):
        T_l2s_comp = apply_angle_compensation(T_lidar_to_sensing, pitch, roll, yaw)
        T_l2c = T_sensing_to_cam @ T_l2s_comp
        scores = []
        first_frame_result = None
        for img, pc, shape in frame_data:
            pts_2d, depths, n = project_points(pc, T_l2c, K, shape)
            s = compute_edge_alignment_score(img, pts_2d, depths)
            scores.append(s)
            if first_frame_result is None:
                first_frame_result = (pts_2d, depths, n)
        avg_score = np.mean(scores) if scores else 0.0
        return avg_score, first_frame_result[0], first_frame_result[1], first_frame_result[2]

    for pitch in angles:
        score, pts_2d, depths, n = _eval(pitch, 0, 0)
        results[(pitch, 0, 0)] = (score, pts_2d, depths, n)
        if score > best_score:
            best_score = score
            best_comp = (pitch, 0, 0)

    for roll in angles:
        if roll == 0:
            continue
        score, pts_2d, depths, n = _eval(best_comp[0], roll, 0)
        results[(best_comp[0], roll, 0)] = (score, pts_2d, depths, n)
        if score > best_score:
            best_score = score
            best_comp = (best_comp[0], roll, 0)

    for yaw in angles:
        if yaw == 0:
            continue
        score, pts_2d, depths, n = _eval(best_comp[0], best_comp[1], yaw)
        results[(best_comp[0], best_comp[1], yaw)] = (score, pts_2d, depths, n)
        if score > best_score:
            best_score = score
            best_comp = (best_comp[0], best_comp[1], yaw)

    return best_comp, results


def main():
    parser = argparse.ArgumentParser(description='快速验证 LiDAR 外参匹配度')
    parser.add_argument('trip_dir', type=str, help='行程目录路径')
    parser.add_argument('--camera', type=str, default='traffic_2', help='相机名称')
    parser.add_argument('--n_frames', type=int, default=3, help='采样帧数')
    parser.add_argument('--compensate', action='store_true', help='执行角度补偿搜索')
    parser.add_argument('--grid', type=float, default=0.5, help='搜索步长 (度)')
    parser.add_argument('--grid_range', type=float, default=1.5, help='搜索范围 (度)')
    parser.add_argument('--apply-pitch', type=float, default=None, help='手动 pitch 补偿 (度)')
    parser.add_argument('--apply-roll', type=float, default=None, help='手动 roll 补偿 (度)')
    parser.add_argument('--apply-yaw', type=float, default=None, help='手动 yaw 补偿 (度)')
    parser.add_argument('--output', type=str, default=None, help='输出目录 (默认: trip_dir/verify_extrinsic)')
    parser.add_argument('--lidar_topic', type=str,
                        default='/sensors/lidar/combined_point_cloud_proto')
    args = parser.parse_args()

    trip_dir = Path(args.trip_dir)
    if not trip_dir.exists():
        print(f"行程目录不存在: {trip_dir}")
        sys.exit(1)

    output_dir = Path(args.output) if args.output else trip_dir / 'verify_extrinsic'
    output_dir.mkdir(parents=True, exist_ok=True)

    trip_name = trip_dir.name
    print(f"\n{'='*70}")
    print(f"快速验证 LiDAR 外参: {trip_name}")
    print(f"相机: {args.camera}")
    print(f"{'='*70}")

    t0 = time.time()

    # --- 1. 解析配置 ---
    configs_cam_path = trip_dir / 'configs' / 'cameras.cfg'
    configs_lidar_path = trip_dir / 'configs' / 'lidars.cfg'
    model_cam_path = trip_dir / 'model' / 'cameras.cfg'
    model_lidar_path = trip_dir / 'model' / 'lidars.cfg'

    has_model = model_cam_path.exists() and model_lidar_path.exists()

    print(f"\n--- 解析配置文件 ---")
    configs_cams = parse_cameras_cfg(str(configs_cam_path))
    configs_lidar = parse_lidars_cfg(str(configs_lidar_path))

    if args.camera not in configs_cams:
        avail = list(configs_cams.keys())
        print(f"相机 '{args.camera}' 不在 cameras.cfg 中。可用: {avail}")
        sys.exit(1)

    configs_cam = configs_cams[args.camera]
    print(f"  configs/cameras.cfg: {args.camera} OK")
    print(f"  configs/lidars.cfg: OK")

    model_cam, model_lidar = None, None
    if has_model:
        model_cams = parse_cameras_cfg(str(model_cam_path))
        model_lidar = parse_lidars_cfg(str(model_lidar_path))
        model_cam = model_cams.get(args.camera)
        print(f"  model/cameras.cfg: {args.camera} {'OK' if model_cam else 'NOT FOUND'}")
        print(f"  model/lidars.cfg: OK")

    # --- 2. 构建变换矩阵 ---
    T_l2c_cfg, T_l2s_cfg, K_cfg = build_transforms(configs_cam, configs_lidar)
    m1, m2, K_undist = init_undistortion(configs_cam, K_cfg)
    K_use = K_undist

    dist = configs_cam.get('distortion', {})
    print(f"\n  Undistortion: model={dist.get('model_type', 'unknown')}, "
          f"k1={dist.get('k1', 0):.6f}, k2={dist.get('k2', 0):.6f}")
    print(f"  K_orig: fx={K_cfg[0,0]:.1f}, fy={K_cfg[1,1]:.1f}, cx={K_cfg[0,2]:.1f}, cy={K_cfg[1,2]:.1f}")
    print(f"  K_undist: fx={K_use[0,0]:.1f}, fy={K_use[1,1]:.1f}, cx={K_use[0,2]:.1f}, cy={K_use[1,2]:.1f}")
    print(f"  Undist maps: {'ready' if m1 is not None else 'skipped (D=0)'}")

    if has_model and model_cam:
        T_l2c_mdl, T_l2s_mdl, K_mdl = build_transforms(model_cam, model_lidar)
        compare_extrinsics(configs_cam, configs_lidar, model_cam, model_lidar)

    # --- 3. 提取样本帧 ---
    print(f"\n--- 从 bag 提取样本帧 (最多 {args.n_frames} 帧) ---")
    frames = extract_sample_frames(trip_dir, args.camera, args.n_frames, args.lidar_topic)
    if not frames:
        print("未能提取到任何帧，请检查 bag 文件和 topic 名称")
        sys.exit(1)
    print(f"  成功提取 {len(frames)} 帧")

    # --- 4. 生成投影可视化 ---
    print(f"\n--- 生成投影可视化 ---")

    for fi, (image_raw, pointcloud) in enumerate(frames):
        if m1 is not None:
            image = cv2.remap(image_raw, m1, m2, cv2.INTER_LINEAR)
        else:
            image = image_raw

        img_shape = image.shape[:2]
        n_cols = 3 if (has_model and model_cam) else 2
        has_manual = (args.apply_pitch is not None or args.apply_roll is not None
                      or args.apply_yaw is not None)
        if has_manual:
            n_cols += 1

        fig, axes = plt.subplots(1, n_cols, figsize=(8 * n_cols, 6), dpi=120)
        if n_cols == 1:
            axes = [axes]

        col = 0

        # configs 投影
        pts_2d, depths, n_total = project_points(pointcloud, T_l2c_cfg, K_use, img_shape)
        render_projection(axes[col], image, pts_2d, depths,
                         f'configs (online-calib)', n_total)
        col += 1

        # model 投影
        if has_model and model_cam:
            m1_m, m2_m, K_mdl_undist = init_undistortion(model_cam, K_mdl)
            pts_2d_m, depths_m, n_m = project_points(pointcloud, T_l2c_mdl, K_mdl_undist, img_shape)
            render_projection(axes[col], image, pts_2d_m, depths_m,
                             f'model (factory)', n_m)
            col += 1

        # 手动补偿投影
        if has_manual:
            pitch_c = args.apply_pitch or 0.0
            roll_c = args.apply_roll or 0.0
            yaw_c = args.apply_yaw or 0.0
            T_l2s_comp = apply_angle_compensation(T_l2s_cfg, pitch_c, roll_c, yaw_c)
            T_sensing_to_cam = np.linalg.inv(
                np.eye(4, dtype=float).__setitem__((slice(3), slice(3)), R.from_quat(configs_cam['orientation']).as_matrix()) or np.eye(4))

            T_cam_to_sensing = np.eye(4)
            T_cam_to_sensing[:3, :3] = R.from_quat(configs_cam['orientation']).as_matrix()
            T_cam_to_sensing[:3, 3] = configs_cam['position']
            T_s2c = np.linalg.inv(T_cam_to_sensing)

            T_l2c_comp = T_s2c @ T_l2s_comp
            pts_2d_c, depths_c, n_c = project_points(pointcloud, T_l2c_comp, K_use, img_shape)
            render_projection(axes[col], image, pts_2d_c, depths_c,
                             f'Comp P={pitch_c:.1f} R={roll_c:.1f} Y={yaw_c:.1f}', n_c)
            col += 1

        # 原图
        axes[col].imshow(image)
        axes[col].set_title('Original (no projection)', fontsize=9)
        axes[col].axis('off')

        fig.suptitle(f'{trip_name} | Frame {fi} | {args.camera} | '
                     f'PointCloud {len(pointcloud)} pts', fontsize=11)
        plt.tight_layout()
        out_path = output_dir / f'frame_{fi:02d}_projection.png'
        plt.savefig(out_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f"  保存: {out_path}")

    # --- 5. 角度补偿搜索 ---
    if args.compensate:
        print(f"\n--- 角度补偿网格搜索 (步长={args.grid}°, 范围=±{args.grid_range}°, "
              f"联合 {len(frames)} 帧) ---")

        T_cam_to_sensing = np.eye(4)
        T_cam_to_sensing[:3, :3] = R.from_quat(configs_cam['orientation']).as_matrix()
        T_cam_to_sensing[:3, 3] = configs_cam['position']
        T_s2c = np.linalg.inv(T_cam_to_sensing)

        frame_data = []
        for img_raw, pc in frames:
            img = cv2.remap(img_raw, m1, m2, cv2.INTER_LINEAR) if m1 is not None else img_raw
            frame_data.append((img, pc, img.shape[:2]))
        image = frame_data[0][0]
        img_shape = frame_data[0][2]
        pointcloud = frames[0][1]

        best_comp, results = grid_search_compensation(
            frame_data, T_s2c, T_l2s_cfg,
            K_use, args.grid, args.grid_range)

        print(f"\n  最佳补偿: Pitch={best_comp[0]:.2f}°, Roll={best_comp[1]:.2f}°, Yaw={best_comp[2]:.2f}°")

        boundary_warn = any(abs(abs(c) - args.grid_range) < args.grid * 0.5
                            for c in best_comp if abs(c) > 1e-6)
        if boundary_warn:
            print(f"  *** WARNING: 最佳补偿触及搜索边界 (±{args.grid_range}°), "
                  f"建议增大 --grid_range 重跑 ***")

        # 多帧平均 baseline score
        baseline_scores = []
        for img, pc, shape in frame_data:
            p2d, dep, _ = project_points(pc, T_l2c_cfg, K_use, shape)
            baseline_scores.append(compute_edge_alignment_score(img, p2d, dep))
        baseline_score = np.mean(baseline_scores)

        sorted_results = sorted(results.items(), key=lambda x: -x[1][0])
        best_score_val = sorted_results[0][1][0] if sorted_results else 0
        improvement_pct = ((best_score_val - baseline_score) / max(baseline_score, 1e-6)) * 100

        print(f"\n  Baseline EdgeScore: {baseline_score:.4f} (avg over {len(frame_data)} frames)")
        print(f"  Top-5 补偿组合 (按 EdgeScore 排序):")
        for i, ((p, r, y), (score, _, _, n)) in enumerate(sorted_results[:5]):
            delta = score - baseline_score
            print(f"    {i+1}. P={p:+.1f}° R={r:+.1f}° Y={y:+.1f}° → "
                  f"EdgeScore={score:.4f} (Δ={delta:+.4f}, {delta/max(baseline_score,1e-6)*100:+.1f}%)")

        if improvement_pct < 2.0 and not boundary_warn:
            print(f"\n  结论: 补偿提升不显著 ({improvement_pct:+.1f}%), 原始外参可能已较准确")

        # 生成 Top-3 + 原始对比图 (带边缘叠加)
        fig = plt.figure(figsize=(32, 8), dpi=120)
        gs = GridSpec(1, 4, figure=fig)

        pts_orig = project_points(pointcloud, T_l2c_cfg, K_use, img_shape)
        ax0 = fig.add_subplot(gs[0, 0])
        render_projection(ax0, image, pts_orig[0], pts_orig[1],
                         'configs original', pts_orig[2], show_edge_overlay=True)

        for i, ((p, r, y), (score, pts_2d, depths, n)) in enumerate(sorted_results[:3]):
            ax = fig.add_subplot(gs[0, i + 1])
            render_projection(ax, image, pts_2d, depths,
                             f'Comp P={p:+.1f} R={r:+.1f} Y={y:+.1f}', n,
                             show_edge_overlay=True)

        fig.suptitle(f'{trip_name} | Compensation Search (Edge Alignment Score)',
                     fontsize=12)
        plt.tight_layout()
        out_path = output_dir / 'compensation_search.png'
        plt.savefig(out_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f"  保存: {out_path}")

        # 输出补偿建议
        if abs(best_comp[0]) > 0.1 or abs(best_comp[1]) > 0.1 or abs(best_comp[2]) > 0.1:
            print(f"\n{'='*70}")
            print(f"建议: 在 lidars.cfg 的 sensor_to_lidar orientation 中叠加如下补偿后重新生成数据:")
            print(f"  Pitch 补偿: {best_comp[0]:+.2f}°")
            print(f"  Roll 补偿:  {best_comp[1]:+.2f}°")
            print(f"  Yaw 补偿:   {best_comp[2]:+.2f}°")
            print(f"\n  操作方法:")
            print(f"    1. 用 --apply-pitch {best_comp[0]:.2f} 再次运行确认效果")
            print(f"    2. 确认后修改 lidars.cfg 的 orientation (四元数)")
            print(f"       当前: qx={configs_lidar['orientation'][0]:.9f}, "
                  f"qy={configs_lidar['orientation'][1]:.9f}, "
                  f"qz={configs_lidar['orientation'][2]:.9f}, "
                  f"qw={configs_lidar['orientation'][3]:.9f}")

            R_orig = R.from_quat(configs_lidar['orientation'])
            R_comp = R.from_euler('xyz', [best_comp[1], best_comp[0], best_comp[2]], degrees=True)
            R_new = R_comp * R_orig
            q_new = R_new.as_quat()
            print(f"       补偿后: qx={q_new[0]:.9f}, qy={q_new[1]:.9f}, "
                  f"qz={q_new[2]:.9f}, qw={q_new[3]:.9f}")
            print(f"{'='*70}")
        else:
            print(f"\n  当前外参匹配良好，无需补偿。")

    elapsed = time.time() - t0
    print(f"\n总耗时: {elapsed:.1f}s")
    print(f"输出目录: {output_dir}")


if __name__ == '__main__':
    main()
