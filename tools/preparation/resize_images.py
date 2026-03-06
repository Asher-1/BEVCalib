#!/usr/bin/env python3
"""
预处理脚本：将 4K PNG 图像批量 resize 为训练目标尺寸 JPEG
消除训练时的 4K PNG 解码 + resize 开销（从 ~4.3MB PNG → ~40KB JPEG）

用法:
    python preprocess_resize.py --dataset_root /path/to/data --width 640 --height 360
"""
import os
import sys
import cv2
import json
import argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import time


def resize_single_image(args_tuple):
    """Worker function: resize one image and save as JPEG."""
    src_path, dst_path, target_w, target_h = args_tuple
    try:
        img = cv2.imread(src_path, cv2.IMREAD_COLOR)
        if img is None:
            return src_path, False, "imread failed"
        resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(dst_path, resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return src_path, True, None
    except Exception as e:
        return src_path, False, str(e)


def main():
    parser = argparse.ArgumentParser(description="Resize 4K PNG images to target size JPEG")
    parser.add_argument("--dataset_root", type=str, required=True, help="Dataset root dir")
    parser.add_argument("--width", type=int, default=640, help="Target width")
    parser.add_argument("--height", type=int, default=360, help="Target height")
    parser.add_argument("--workers", type=int, default=32, help="Parallel workers")
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality (0-100)")
    args = parser.parse_args()

    target_w, target_h = args.width, args.height
    resized_dir_name = f"image_2_{target_w}x{target_h}"
    seq_root = os.path.join(args.dataset_root, "sequences")

    if not os.path.isdir(seq_root):
        print(f"ERROR: {seq_root} not found")
        sys.exit(1)

    sequences = sorted([
        d for d in os.listdir(seq_root)
        if os.path.isdir(os.path.join(seq_root, d, "image_2"))
    ])
    print(f"Found {len(sequences)} sequences: {sequences}")

    tasks = []
    for seq in sequences:
        src_dir = os.path.join(seq_root, seq, "image_2")
        dst_dir = os.path.join(seq_root, seq, resized_dir_name)
        os.makedirs(dst_dir, exist_ok=True)

        for fname in sorted(os.listdir(src_dir)):
            if not fname.endswith(".png"):
                continue
            src_path = os.path.join(src_dir, fname)
            dst_fname = fname.replace(".png", ".jpg")
            dst_path = os.path.join(dst_dir, dst_fname)
            if os.path.exists(dst_path):
                continue
            tasks.append((src_path, dst_path, target_w, target_h))

    total = len(tasks)
    if total == 0:
        print(f"All images already resized in {resized_dir_name}/. Nothing to do.")
        # Save metadata anyway
        _save_metadata(seq_root, sequences, resized_dir_name, target_w, target_h)
        return

    print(f"Resizing {total} images → {target_w}x{target_h} JPEG (quality={args.quality}), workers={args.workers}")
    t0 = time.time()
    done = 0
    failed = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(resize_single_image, t): t for t in tasks}
        for future in as_completed(futures):
            src_path, ok, err = future.result()
            done += 1
            if not ok:
                failed += 1
                print(f"  FAIL: {src_path}: {err}")
            if done % 2000 == 0 or done == total:
                elapsed = time.time() - t0
                rate = done / elapsed
                eta = (total - done) / rate if rate > 0 else 0
                print(f"  [{done}/{total}] {rate:.1f} img/s, elapsed={elapsed:.0f}s, ETA={eta:.0f}s")

    elapsed = time.time() - t0
    print(f"\nDone! {done - failed}/{total} images resized in {elapsed:.1f}s ({(done-failed)/elapsed:.1f} img/s)")
    if failed > 0:
        print(f"  {failed} images failed")

    _save_metadata(seq_root, sequences, resized_dir_name, target_w, target_h)

    # Print size comparison
    _print_size_comparison(seq_root, sequences, resized_dir_name)


def _save_metadata(seq_root, sequences, resized_dir_name, target_w, target_h):
    """Save metadata JSON alongside the resized images."""
    meta = {
        "target_width": target_w,
        "target_height": target_h,
        "resized_dir_name": resized_dir_name,
        "format": "jpeg",
        "quality": 95,
    }
    meta_path = os.path.join(seq_root, f"{resized_dir_name}_meta.json")
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved to: {meta_path}")


def _print_size_comparison(seq_root, sequences, resized_dir_name):
    """Print file size comparison for first sequence."""
    seq = sequences[0]
    orig_dir = os.path.join(seq_root, seq, "image_2")
    resized_dir = os.path.join(seq_root, seq, resized_dir_name)

    orig_files = [f for f in os.listdir(orig_dir) if f.endswith(".png")][:10]
    total_orig = sum(os.path.getsize(os.path.join(orig_dir, f)) for f in orig_files)
    resized_files = [f.replace(".png", ".jpg") for f in orig_files]
    total_resized = sum(
        os.path.getsize(os.path.join(resized_dir, f))
        for f in resized_files if os.path.exists(os.path.join(resized_dir, f))
    )

    if total_orig > 0 and total_resized > 0:
        n = len(orig_files)
        print(f"\nSize comparison (seq {seq}, {n} samples):")
        print(f"  Original PNG:  avg {total_orig/n/1024:.0f} KB/file")
        print(f"  Resized JPEG:  avg {total_resized/n/1024:.0f} KB/file")
        print(f"  Reduction:     {total_resized/total_orig*100:.1f}% ({total_orig/total_resized:.1f}x smaller)")


if __name__ == "__main__":
    main()
