#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_bag_calibration.py
======================

Production-style BEVCalib LiDAR-to-camera calibration from raw rosbag data.

Pipeline (per trip):
  1) Locate bags + configs (trip layout or explicit bag-list file).
  2) Reuse ``BEVCalibDatasetPreparer`` to extract & sync JPEG + PointCloud2 + poses
     into a temporary KITTI-style tree (images are undistorted, point clouds aligned).
  3) Optionally subsample frames in time / distance along the synced sequence.
  4) Batch inference through ``utils.bevcalib_inference.load_bevcalib_inference``.
  5) Robust temporal aggregation via ``TemporalCalibrationAggregator`` (axis-angle median).
  6) Persist matrices, textual reports, Markdown summary, and side-by-side projection PNGs.

The script is intended to be invoked from the BEVCalib repository root::

    python run_bag_calibration.py --input_file trips.txt \\
        --ckpt_path path/to/model.pth --output_dir outputs/run1

Requirements: Python 3.6+ (avoid walrus operator and f-string debugging ``f"{x=}"``).

Optional heavy dependencies emit ``ImportError`` hints when missing (torch, rosbags, ...).
"""

from __future__ import print_function

# =============================================================================
# Section: Matplotlib backend (must be before pyplot imports anywhere)
# =============================================================================
import matplotlib

matplotlib.use("Agg")

import argparse
import bisect
import contextlib
import datetime
import io
import json
import logging
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml


# -----------------------------------------------------------------------------
# Executable directory = BEVCalib repository root when script lives alongside
# ``tools/``, ``utils/``, and ``kitti-bev-calib/``.
# -----------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

_PREP_DIR = os.path.join(_SCRIPT_DIR, "tools", "preparation")
if _PREP_DIR not in sys.path:
    sys.path.insert(0, _PREP_DIR)

_KITTI_DIR = os.path.join(_SCRIPT_DIR, "kitti-bev-calib")
if _KITTI_DIR not in sys.path:
    sys.path.insert(0, _KITTI_DIR)

import numpy as np

_LOGGER = logging.getLogger("run_bag_calibration")


# =============================================================================
# Section: Torch / scipy / cv2 (soft-fail diagnostics)
# =============================================================================
def _require_torch():
    """Import torch lazily so ``--help`` works in CPU-only shells."""
    try:
        import torch
    except ImportError:
        raise ImportError(
            "PyTorch is required for inference. Install torch matching your CUDA version."
        )
    return torch


def _require_scipy_rot():
    from scipy.spatial.transform import Rotation as ScipyRot

    return ScipyRot


def _require_cv2():
    try:
        import cv2
    except ImportError:
        raise ImportError("OpenCV (cv2) is required for image IO and visualization.")
    return cv2


# =============================================================================
# Section: Tee / logging helpers
# =============================================================================
class StreamTee(object):
    """
    Duplicate writes to multiple text streams.

    Used to mirror ``stdout`` into ``calibration.log`` while preserving console output.
    Not thread-safe for arbitrary parallel writes; guarded usage per trip recommended.
    """

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


@contextlib.contextmanager
def tee_stdout_stderr(log_file_path):
    """
    Context manager attaching a tee to stdout/stderr writing to ``log_file_path``.

    Args:
        log_file_path (str): Path to append UTF-8 log lines.
    """
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    log_file_path = os.path.abspath(log_file_path)

    lf = io.open(log_file_path, "a", encoding="utf-8")

    prev_out = sys.stdout
    prev_err = sys.stderr
    tee_out = StreamTee(prev_out, lf)
    tee_err = StreamTee(prev_err, lf)
    sys.stdout = tee_out
    sys.stderr = tee_err

    yield log_file_path

    sys.stdout.flush()
    sys.stderr.flush()
    lf.flush()
    lf.close()

    sys.stdout = prev_out
    sys.stderr = prev_err


# =============================================================================
# Section: Preparation imports (configs + dataset preparer)
# =============================================================================
try:
    from prepare_custom_dataset import ConfigParser, BEVCalibDatasetPreparer
except ImportError:
    ConfigParser = None
    BEVCalibDatasetPreparer = None


try:
    from utils.bevcalib_inference import (
        TemporalCalibrationAggregator,
        load_bevcalib_inference,
    )
except ImportError:
    TemporalCalibrationAggregator = None
    load_bevcalib_inference = None


def _lazy_viz_funcs():
    """
    Lazy import visualization utilities from ``kitti-bev-calib/visualization.py``.
    Returns:
        tuple: (project_points_to_image, render_projected_points, compute_pose_errors)
    """
    try:
        import importlib.util

        viz_path = os.path.join(_KITTI_DIR, "visualization.py")
        spec = importlib.util.spec_from_file_location("bevcalib_run_viz", viz_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.project_points_to_image, mod.render_projected_points, mod.compute_pose_errors
    except Exception as exc:
        _LOGGER.warning("Visualization unavailable: %s", exc)
        return None, None, None


def _lazy_bev_bounds():
    """Return (xbound, ybound, zbound) tuples from ``bev_settings``."""
    from bev_settings import xbound, ybound, zbound

    return xbound, ybound, zbound


# =============================================================================
# Section: Quaternion / transforms
# =============================================================================
def compute_T_lidar_to_cam(camera_config, lidar_config):
    """
    Chain LiDAR→Sensing→Camera transforms parsed from configs.

    ``cameras.cfg`` stores ``sensor_to_cam`` which is interpreted as Camera→Sensing::

        T_cam_to_sensing = build_from_quat_pos(...)
        T_sensing_to_cam = inv(T_cam_to_sensing)

    ``lidars.cfg`` stores ``sensor_to_lidar`` as LiDAR→Sensing::

        T_lidar_to_sensing = build_from_quat_pos(...)
        T_lidar_to_cam = T_sensing_to_cam @ T_lidar_to_sensing

    Args:
        camera_config (dict): Parsed camera block (position, orientation quat xyzw).
        lidar_config (dict): Parsed lidar block with position/orientation scalars.

    Returns:
        np.ndarray: (4,4) float64 LiDAR→Camera matrix.
    """
    ScipyRot = _require_scipy_rot()

    T_cam_to_sensing = np.eye(4, dtype=np.float64)
    T_cam_to_sensing[:3, :3] = ScipyRot.from_quat(camera_config["orientation"]).as_matrix()
    T_cam_to_sensing[:3, 3] = camera_config["position"]
    T_sensing_to_cam = np.linalg.inv(T_cam_to_sensing)

    T_lidar_to_sensing = np.eye(4, dtype=np.float64)
    T_lidar_to_sensing[:3, :3] = ScipyRot.from_quat(lidar_config["orientation"]).as_matrix()
    T_lidar_to_sensing[:3, 3] = lidar_config["position"]

    return T_sensing_to_cam @ T_lidar_to_sensing


def rotation_geodesic_angle_deg(R_a, R_b):
    """Geodesic angle (degrees) between two rotation matrices."""
    R_delta = np.dot(R_a.T, R_b)
    trace = np.clip((np.trace(R_delta) - 1.0) * 0.5, -1.0, 1.0)
    return math.degrees(math.acos(trace))


def T_to_calibration_metrics(T_orig, T_cal):
    """
    Compare two LiDAR→Camera transforms.

    Returns:
        dict with keys: rot_geodesic_deg, trans_l2_m, delta_T (4x4),
        roll_delta_deg, pitch_delta_deg, yaw_delta_deg.
    """
    ScipyRot = _require_scipy_rot()
    R_o, t_o = T_orig[:3, :3], T_orig[:3, 3]
    R_c, t_c = T_cal[:3, :3], T_cal[:3, 3]

    R_delta = R_o.T @ R_c
    rpy_delta = ScipyRot.from_matrix(R_delta).as_euler("xyz", degrees=True)

    return {
        "rot_geodesic_deg": float(rotation_geodesic_angle_deg(R_o, R_c)),
        "trans_l2_m": float(np.linalg.norm(t_c - t_o)),
        "delta_T": np.linalg.inv(T_orig) @ T_cal,
        "roll_delta_deg": float(rpy_delta[0]),
        "pitch_delta_deg": float(rpy_delta[1]),
        "yaw_delta_deg": float(rpy_delta[2]),
    }


# =============================================================================
# Section: Input discovery (bag_list / trips / auto)
# =============================================================================
def detect_input_format(input_path):
    """
    Heuristic auto-detect for ``bag_list`` vs ``trips`` text files.

    Rules:
      * If any non-comment line contains ``.bag`` token → ``bag_list``.
      * Else → ``trips``.
    """
    with io.open(input_path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ".bag" in line:
                return "bag_list"
    return "trips"


def parse_trips_file(input_path):
    """
    Parse ``trips.txt`` (one trip name per line, ``#`` comments allowed).

    Returns:
        list of str: Trip identifiers in file order.
    """
    trips = []
    with io.open(input_path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            trips.append(line.split()[0])
    return trips


class StreamingTripDownloader(object):
    """
    Streaming trip downloader inspired by C++ DPBag View pattern.

    Instead of downloading everything upfront, this class:
      1) Downloads configs synchronously (small, instant).
      2) Lists remote bags and selects a time-distributed subset.
      3) Returns immediately so calibration can prepare.
      4) Downloads bags in background threads into a staging directory.
      5) Exposes a stop_flag so the caller can halt downloads once enough frames
         are extracted (early termination, mirroring ``is_data_enough()`` in C++).

    Usage in calibrate_trip:
        downloader = StreamingTripDownloader(trip_name, trips_base)
        downloader.prepare()           # sync: configs + bag list
        downloader.start_downloads()   # async: bags stream in background
        # ... preparer processes bags as they land ...
        downloader.stop()              # signal background threads to stop
    """

    def __init__(self, trip_name, trips_base, initial_bag_groups=10,
                 download_workers=8, max_bag_groups=None):
        if max_bag_groups is not None:
            initial_bag_groups = max_bag_groups
        self.trip_name = trip_name
        self.trips_base = trips_base
        self.initial_bag_groups = initial_bag_groups
        self.download_workers = download_workers

        self.trip_dir = os.path.join(trips_base, trip_name)
        self.bag_staging = os.path.join(self.trip_dir, "bags", "important")
        self.config_dir = os.path.join(self.trip_dir, "configs")

        self._stop = threading.Event()
        self._download_queue = []
        self._downloaded_count = 0
        self._download_lock = threading.Lock()
        self._executor = None
        self._all_paired_slots = []
        self._queued_slot_count = 0

        self.drfile_bin = shutil.which("drfile")
        if self.drfile_bin is None:
            raise RuntimeError("drfile not found on PATH. Install drfile to use remote mode.")

    @property
    def downloaded_count(self):
        with self._download_lock:
            return self._downloaded_count

    @property
    def has_unqueued_slots(self):
        return self._queued_slot_count < len(self._all_paired_slots)

    @property
    def all_downloaded(self):
        with self._download_lock:
            queue_len = len(self._download_queue)
        return self.downloaded_count >= queue_len and not self.has_unqueued_slots

    def _queue_slots(self, slots):
        """Add bags for the given time slots to the download queue (thread-safe).

        Download order per slot: Light (small, fast) -> Heavy (images) -> Medium (pointcloud).
        Slots are interleaved so each slot gets complete data ASAP, enabling
        earlier extraction rounds in streaming mode.
        """
        with self._download_lock:
            prev_len = len(self._download_queue)
            for slot in slots:
                for tg, name in sorted(slot, key=lambda x: {"Light_Topic_Group": 0,
                                                             "Heavy_Topic_Group": 1,
                                                             "Medium_Topic_Group": 2}.get(x[0], 9)):
                    local_dir = os.path.join(self.bag_staging, tg)
                    os.makedirs(local_dir, exist_ok=True)
                    uri = "trip:/{}/bags/important/{}/{}".format(
                        self.trip_name, tg, name)
                    self._download_queue.append((uri, local_dir, name))
            return len(self._download_queue) - prev_len

    def request_more_bags(self, n_groups=5):
        """Queue additional time slots for download. Call when more data is needed."""
        if self._queued_slot_count >= len(self._all_paired_slots):
            return 0
        start = self._queued_slot_count
        end = min(start + n_groups, len(self._all_paired_slots))
        new_slots = self._all_paired_slots[start:end]
        n_new = self._queue_slots(new_slots)
        self._queued_slot_count = end
        if self._executor and n_new > 0:
            with self._download_lock:
                new_items = list(self._download_queue[-n_new:])
            for uri, local_dir, name in new_items:
                self._executor.submit(self._download_one, uri, local_dir, name)
        print("[stream] Requested {} more bag groups ({} new bags), total queued: {}/{}".format(
            end - start, n_new, self._queued_slot_count, len(self._all_paired_slots)))
        return n_new

    def _ensure_model_folder(self):
        """Download model/lidars.cfg + model/cameras.cfg for IAE and GT comparison.

        Both files are required by ``_try_load_gt_extrinsic``.  Only downloads
        text cfg files to avoid failures from corrupted/missing binary files
        (e.g. ground.bin returning 404).  Resolves configs_onboard LINK first.
        """
        model_dir = os.path.join(self.trip_dir, "model")
        needed = (
            ("lidars.cfg", "lidars.cfg"),
            ("cameras.cfg", "cameras.cfg"),
        )
        if all(os.path.isfile(os.path.join(model_dir, fname)) and
               os.path.getsize(os.path.join(model_dir, fname)) > 10
               for _, fname in needed):
            print("[stream] Model lidars.cfg + cameras.cfg already present for {}".format(
                self.trip_name))
            return

        os.makedirs(model_dir, exist_ok=True)
        target_ns, target_path = self._resolve_configs_onboard_link()
        rel_paths = [rel for rel, _ in needed]

        def _attempt_uris(rel_path):
            uris = []
            if target_ns and target_path:
                uris.append("{}:{}/sensors/model/{}".format(
                    target_ns, target_path, rel_path))
            uris.append("trip:/{}/configs_onboard/sensors/model/{}".format(
                self.trip_name, rel_path))
            uris.append("trip:/{}/configs/sensors/model/{}".format(
                self.trip_name, rel_path))
            return uris

        for rel_path, fname in needed:
            target = os.path.join(model_dir, fname)
            if os.path.isfile(target) and os.path.getsize(target) > 10:
                continue
            for uri in _attempt_uris(rel_path):
                _drfile_download_quiet(self.drfile_bin, uri, model_dir)
                if os.path.isfile(target) and os.path.getsize(target) > 10:
                    print("[stream] Model {} ready for {} (from {})".format(
                        fname, self.trip_name, uri.split(":")[0]))
                    break
            else:
                print("[stream] Model {} not available for {} (tried {} paths)".format(
                    fname, self.trip_name, len(_attempt_uris(rel_path))))

        have_lidars = os.path.isfile(os.path.join(model_dir, "lidars.cfg"))
        have_cams = os.path.isfile(os.path.join(model_dir, "cameras.cfg"))
        if not (have_lidars and have_cams):
            print("[stream] Model folder incomplete for {} "
                  "(lidars.cfg={}, cameras.cfg={}; IAE/GT may use fallback)".format(
                      self.trip_name, have_lidars, have_cams))

    def _resolve_configs_onboard_link(self):
        """Resolve configs_onboard LINK to its target namespace and path.

        drfile 2.38+ writes both the request log and response JSON to stdout,
        so we must locate the *response* JSON (the one containing ``"target"``)
        rather than naively taking the first ``{``.
        """
        try:
            proc = subprocess.Popen(
                [self.drfile_bin, "head", "--format", "json",
                 "trip:/{}/configs_onboard".format(self.trip_name)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                universal_newlines=True,
            )
            stdout, _ = proc.communicate(timeout=15)
            if proc.returncode == 0 and '"target"' in stdout:
                target_pos = stdout.find('"target"')
                if target_pos < 0:
                    return None, None
                json_start = stdout.rfind("{", 0, target_pos)
                brace_depth = 0
                json_end = -1
                for i in range(json_start, len(stdout)):
                    if stdout[i] == "{":
                        brace_depth += 1
                    elif stdout[i] == "}":
                        brace_depth -= 1
                        if brace_depth == 0:
                            json_end = i + 1
                            break
                if json_end > json_start:
                    try:
                        data = json.loads(stdout[json_start:json_end])
                        if isinstance(data, list):
                            data = data[0]
                        target = data.get("target", {})
                        ns = target.get("namespace")
                        path = target.get("path")
                        if ns and path:
                            print("[stream] Resolved configs_onboard LINK -> {}:{}".format(ns, path))
                            return ns, path
                    except json.JSONDecodeError:
                        pass
        except Exception:
            pass
        return None, None

    def prepare(self):
        """Sync step: download configs + list remote bags + verify cache completeness."""
        os.makedirs(self.trip_dir, exist_ok=True)

        has_configs = (os.path.isdir(self.config_dir)
                       and (os.path.isfile(os.path.join(self.config_dir, "lidars.cfg"))
                            or os.path.isfile(os.path.join(self.config_dir, "cameras.cfg"))))
        if not has_configs:
            _drfile_download_quiet(self.drfile_bin,
                                   "trip:/{}/configs".format(self.trip_name),
                                   os.path.join(self.trip_dir, "."))
            if not os.path.isdir(self.config_dir):
                raise RuntimeError("Failed to download configs for {}".format(self.trip_name))
        print("[stream] Configs ready for {}".format(self.trip_name))

        self._ensure_model_folder()

        bag_groups = {}
        for tg in ("Heavy_Topic_Group", "Medium_Topic_Group", "Light_Topic_Group"):
            remote_dir = "trip:/{}/bags/important/{}".format(self.trip_name, tg)
            names = _drfile_list_bags(self.drfile_bin, remote_dir)
            bag_groups[tg] = sorted(names)

        local_bags = {}
        for tg in ("Heavy_Topic_Group", "Medium_Topic_Group", "Light_Topic_Group"):
            tg_dir = os.path.join(self.bag_staging, tg)
            if os.path.isdir(tg_dir):
                local_bags[tg] = {
                    f for f in os.listdir(tg_dir)
                    if f.endswith(".bag") and not f.startswith(".__tmp__")
                       and os.path.getsize(os.path.join(tg_dir, f)) > 1000
                }
            else:
                local_bags[tg] = set()

        total_remote = sum(len(v) for v in bag_groups.values())
        total_local = sum(len(v) for v in local_bags.values())
        missing = {}
        for tg, remote_names in bag_groups.items():
            diff = set(remote_names) - local_bags.get(tg, set())
            if diff:
                missing[tg] = diff
        total_missing = sum(len(v) for v in missing.values())

        if total_missing > 0:
            print("[stream] Cache verification: {}/{} bags present, {} missing — will download".format(
                total_local, total_remote, total_missing))
        else:
            print("[stream] Cache verified: {}/{} bags complete for {}".format(
                total_local, total_remote, self.trip_name))

        def _ts_key(name):
            parts = name.split(".")
            return parts[0] if parts else name

        group_maps = {}
        for tg, names in bag_groups.items():
            m = {}
            for n in names:
                m[_ts_key(n)] = (tg, n)
            group_maps[tg] = m

        all_ts = sorted(set(
            k for m in group_maps.values() for k in m.keys()
        ))
        paired_slots = []
        for ts in all_ts:
            slot = []
            for tg in ("Heavy_Topic_Group", "Medium_Topic_Group", "Light_Topic_Group"):
                if ts in group_maps[tg]:
                    slot.append(group_maps[tg][ts])
            paired_slots.append(slot)

        self._all_paired_slots = paired_slots
        initial_count = min(self.initial_bag_groups, len(paired_slots))
        initial_slots = paired_slots[:initial_count]
        total_bags = sum(len(s) for s in initial_slots)
        h_cnt = len(bag_groups.get("Heavy_Topic_Group", []))
        m_cnt = len(bag_groups.get("Medium_Topic_Group", []))
        l_cnt = len(bag_groups.get("Light_Topic_Group", []))
        self._queue_slots(initial_slots)
        self._queued_slot_count = initial_count
        print("[stream] Initial {}/{} time slots ({} bags queued, H={} M={} L={})".format(
            self._queued_slot_count, len(paired_slots), total_bags,
            h_cnt, m_cnt, l_cnt))

    def start_downloads(self):
        """Start background download threads. Returns immediately."""
        if not self._download_queue:
            print("[stream] No bags to download (using cached data)")
            return
        print("[stream] Starting {} concurrent downloads ({} bags queued)".format(
            self.download_workers, len(self._download_queue)))
        self._executor = ThreadPoolExecutor(max_workers=self.download_workers)
        for uri, local_dir, name in self._download_queue:
            self._executor.submit(self._download_one, uri, local_dir, name)

    def _download_one(self, uri, local_dir, name):
        """Download a single bag. Respects stop_flag."""
        if self._stop.is_set():
            return
        local_path = os.path.join(local_dir, name)
        if os.path.isfile(local_path) and os.path.getsize(local_path) > 1000:
            with self._download_lock:
                self._downloaded_count += 1
            return
        try:
            proc = subprocess.Popen(
                [self.drfile_bin, "download", uri, local_dir],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                universal_newlines=True, bufsize=1,
            )
            for line in proc.stdout:
                if self._stop.is_set():
                    proc.terminate()
                    return
            proc.wait()
            if proc.returncode == 0 and os.path.isfile(local_path):
                with self._download_lock:
                    self._downloaded_count += 1
                    cnt = self._downloaded_count
                print("[stream] Bag landed [{}/{}]: {}".format(
                    cnt, len(self._download_queue), name))
        except Exception as exc:
            print("[stream] WARNING: download failed for {}: {}".format(name, exc))

    def wait_for_min_bags(self, min_count=4, timeout=300):
        """Block until at least min_count bags have been downloaded."""
        t0 = time.time()
        while self.downloaded_count < min_count:
            if time.time() - t0 > timeout:
                print("[stream] WARNING: timeout waiting for min bags ({}/{})".format(
                    self.downloaded_count, min_count))
                break
            if self._stop.is_set():
                break
            time.sleep(1)

    def wait_for_completion(self, timeout=600):
        """Block until all queued bags have been downloaded (or timeout)."""
        total = len(self._download_queue)
        if total == 0:
            return
        t0 = time.time()
        while self.downloaded_count < total:
            if time.time() - t0 > timeout:
                print("[stream] WARNING: timeout waiting for all bags ({}/{})".format(
                    self.downloaded_count, total))
                break
            if self._stop.is_set():
                break
            time.sleep(2)
        if self._executor:
            self._executor.shutdown(wait=True)
        self._stop.set()

    def stop(self):
        """Signal all background downloads to stop."""
        self._stop.set()
        if self._executor is not None:
            self._executor.shutdown(wait=False)
        print("[stream] Downloads stopped ({} bags landed)".format(self.downloaded_count))

    def get_bag_root(self):
        """Return the directory where bags are landing (for preparer)."""
        return os.path.join(self.trip_dir, "bags", "important")


def download_remote_trip(trip_name, trips_base, max_bag_groups=20):
    """
    Download a remote trip via ``drfile`` (streaming mode).

    Returns:
        str: Absolute path to the local trip directory.
    """
    downloader = StreamingTripDownloader(trip_name, trips_base, max_bag_groups)
    downloader.prepare()
    if downloader._download_queue:
        downloader.start_downloads()
        downloader.wait_for_min_bags(min_count=4, timeout=300)
        _ACTIVE_DOWNLOADERS[trip_name] = downloader
    else:
        # Cached trip — use incremental local gate instead of full-bag extract
        register_local_incremental_gate(trip_name, downloader.trip_dir,
                                        initial_bag_groups=downloader.initial_bag_groups)
    return downloader.trip_dir


_ACTIVE_DOWNLOADERS = {}


def _drfile_download_quiet(drfile_bin, remote_uri, local_dest):
    """Download via drfile, only print completion or errors."""
    proc = subprocess.Popen(
        [drfile_bin, "download", remote_uri, local_dest],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        universal_newlines=True, bufsize=1,
    )
    for _ in proc.stdout:
        pass
    proc.wait()
    if proc.returncode != 0:
        print("[stream] WARNING: download {} rc={}".format(remote_uri, proc.returncode))


def _drfile_list_bags(drfile_bin, remote_dir):
    """List .bag files in a remote drfile directory. Paginates if needed."""
    bags = []
    page = 0
    page_size = 200
    while True:
        try:
            result = subprocess.run(
                [drfile_bin, "list", remote_dir, "--size", str(page_size),
                 "--format", "json"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                universal_newlines=True, timeout=60,
            )
        except Exception as exc:
            print("[remote] WARNING: list failed for {}: {}".format(remote_dir, exc))
            break

        raw = result.stdout
        data = _extract_drfile_json(raw)
        if data is None:
            for line in raw.splitlines():
                m = re.search(r'/([^\s/]+\.bag)', line)
                if m:
                    bags.append(m.group(1))
            break

        files = data.get("files", [])
        for item in files:
            name = item.get("name", "")
            if name.endswith(".bag"):
                bags.append(name)

        total = data.get("total", 0)
        if len(bags) >= total or not files:
            break
        page += 1

    print("[remote] Listed {} .bag files in {}".format(len(bags), remote_dir.split("/")[-1]))
    return bags


def _extract_drfile_json(raw):
    """Extract the response JSON object from drfile stdout which may contain logs and footers."""
    idx = raw.find('"files"')
    if idx < 0:
        return None
    for i in range(idx, -1, -1):
        if raw[i] == "{":
            depth, j = 0, i
            while j < len(raw):
                if raw[j] == "{":
                    depth += 1
                elif raw[j] == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(raw[i : j + 1])
                        except json.JSONDecodeError:
                            return None
                j += 1
            break
    return None


def _walk_bags(bags_root):
    """Yield .bag file paths under bags_root (skip .__tmp__ partials)."""
    if not os.path.isdir(bags_root):
        return
    for root, _, files in os.walk(bags_root):
        for name in files:
            if name.endswith(".bag") and not name.startswith(".__tmp__"):
                yield os.path.join(root, name)


def parse_bag_list_file(input_path, bags_base):
    """
    Parse ``bag_list.txt`` where *all* lines belong to a single trip.

    Each non-comment line lists multiple ``*.bag`` paths (space separated). Every token
    is joined with ``bags_base`` when not absolute.

    Returns:
        tuple: (trip_name, list of absolute bag paths)
    """
    bag_paths = []
    first_token = None
    with io.open(input_path, "r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            for token in parts:
                if not token.endswith(".bag"):
                    continue
                path = token
                if not os.path.isabs(path):
                    path = os.path.join(bags_base, token)
                bag_paths.append(os.path.abspath(path))
            if first_token is None and parts:
                first_token = parts[0]

    if not bag_paths:
        raise ValueError("No .bag entries found in bag_list file: {}".format(input_path))

    if first_token is None:
        trip_name = "trip_from_bag_list"
    else:
        base = os.path.basename(first_token)
        trip_name = base.split(".")[0]

    return trip_name, bag_paths


def find_trip_directory(trips_base, trip_name):
    """Return ``<trips_base>/<trip_name>`` if it exists."""
    root = os.path.join(trips_base, trip_name)
    if not os.path.isdir(root):
        raise FileNotFoundError("Trip directory not found: {}".format(root))
    return root


def find_trip_config_dir(trip_dir):
    """Return ``<trip_dir>/configs`` path."""
    cfg = os.path.join(trip_dir, "configs")
    if not os.path.isdir(cfg):
        raise FileNotFoundError("configs/ missing under trip: {}".format(trip_dir))
    return cfg


def find_trip_bag_paths(trip_dir):
    """
    Discover ``*.bag`` files for a trip.

    Preference order:
      1) ``bags/important/**/*.bag``
      2) ``bags/**/*.bag``
    """
    important = os.path.join(trip_dir, "bags", "important")
    cands = []
    if os.path.isdir(important):
        for root, _, files in os.walk(important):
            for name in files:
                if name.endswith(".bag") and not name.startswith(".__tmp__"):
                    cands.append(os.path.join(root, name))
    if not cands:
        bags_root = os.path.join(trip_dir, "bags")
        if os.path.isdir(bags_root):
            for root, _, files in os.walk(bags_root):
                for name in files:
                    if name.endswith(".bag") and not name.startswith(".__tmp__"):
                        cands.append(os.path.join(root, name))
    cands.sort()
    if not cands:
        raise FileNotFoundError("No bag files discovered under {}".format(trip_dir))
    return cands


def subsample_bags_for_calibration(bag_paths, target_frames, fps_est=10.0, bag_duration_est=8.0):
    """
    Uniformly subsample bag files to cover roughly ``target_frames`` worth of data
    across the trip's time span.

    Heavy-topic bags contain images (needed), Light/Medium contain poses/pointclouds.
    We select bag *time groups* to maintain topic completeness.

    Args:
        bag_paths (list): All bag paths for the trip.
        target_frames (int): Desired number of frames.
        fps_est (float): Estimated frames per second.
        bag_duration_est (float): Estimated seconds per bag group (~8s per group
            based on typical rosbag recording patterns).

    Returns:
        list: Subset of bag_paths covering the target frame count.
    """
    if len(bag_paths) <= 20:
        return bag_paths

    frames_per_bag_group = fps_est * bag_duration_est
    needed_groups = max(1, int(math.ceil(target_frames / frames_per_bag_group)))

    basenames = sorted(set(os.path.basename(p) for p in bag_paths))
    time_groups = {}
    for name in basenames:
        parts = name.split(".")
        if len(parts) >= 2:
            time_key = parts[0]
        else:
            time_key = name
        time_key = re.sub(r'_(Heavy|Light|Medium|Tiny)_Topic_Group$', '', time_key)
        time_groups.setdefault(time_key, [])

    path_by_base = {}
    for p in bag_paths:
        path_by_base[os.path.basename(p)] = p

    for name, p in path_by_base.items():
        stripped = name.split(".")[0] if "." in name else name
        stripped = re.sub(r'_(Heavy|Light|Medium|Tiny)_Topic_Group$', '', stripped)
        if stripped in time_groups:
            time_groups[stripped].append(p)

    sorted_keys = sorted(time_groups.keys())
    n_groups = len(sorted_keys)

    if n_groups <= needed_groups:
        return bag_paths

    buffer = max(needed_groups, int(needed_groups * 1.5))
    indices = sorted(set(int(round(i)) for i in np.linspace(0, n_groups - 1, num=buffer)))
    selected = []
    for idx in indices:
        selected.extend(time_groups[sorted_keys[idx]])

    _LOGGER.info("Bag subsampling: %d/%d groups selected (%d/%d bags) for ~%d frames",
                 len(indices), n_groups, len(selected), len(bag_paths), target_frames)
    return selected


def _build_bag_time_slots(bag_paths):
    """Group bag paths by time slot (Heavy+Medium+Light per timestamp)."""
    path_by_base = {}
    for p in bag_paths:
        path_by_base[os.path.basename(p)] = os.path.abspath(p)

    time_groups = {}
    for name, p in path_by_base.items():
        stripped = name.split(".")[0] if "." in name else name
        stripped = re.sub(r'_(Heavy|Light|Medium|Tiny)_Topic_Group$', '', stripped)
        time_groups.setdefault(stripped, []).append(p)

    return [time_groups[k] for k in sorted(time_groups.keys())]


class LocalBagGate(object):
    """Incremental local-bag gate: release time-slot groups on demand (no drfile).

    Mimics ``StreamingTripDownloader`` interface for ``_streaming_calibrate_loop``.
    Only symlinks the next N time-slot groups into staging when ``request_more_bags``
    is called — avoids parsing the entire trip's bags upfront.
    """

    def __init__(self, bag_paths, initial_groups=3):
        self._slots = _build_bag_time_slots(bag_paths)
        self._released = 0
        self._initial_groups = max(1, int(initial_groups))
        self._stop = threading.Event()
        self._download_lock = threading.Lock()
        self._pending_paths = []
        self._total_bags = sum(len(s) for s in self._slots)

    @property
    def downloaded_count(self):
        with self._download_lock:
            return self._released

    @property
    def has_unqueued_slots(self):
        return self._released < len(self._slots)

    @property
    def all_downloaded(self):
        return self._released >= len(self._slots)

    def request_more_bags(self, n_groups=3):
        with self._download_lock:
            if self._released >= len(self._slots):
                return 0
            end = min(self._released + int(n_groups), len(self._slots))
            new_paths = []
            for i in range(self._released, end):
                new_paths.extend(self._slots[i])
            self._released = end
            self._pending_paths.extend(new_paths)
            n_new = len(new_paths)
        if n_new > 0:
            print("[stream] Local gate: released groups {}/{} (+{} bags)".format(
                self._released, len(self._slots), n_new))
        return n_new

    def snapshot_new_bags(self, staging, seen_bags):
        """Symlink newly released bags into flat staging (deduped by seen_bags)."""
        os.makedirs(staging, exist_ok=True)
        with self._download_lock:
            pending = list(self._pending_paths)
            self._pending_paths = []
        newly = []
        for full in pending:
            if full in seen_bags:
                continue
            try:
                if os.path.getsize(full) < 1000:
                    continue
            except OSError:
                continue
            name = os.path.basename(full)
            link_dst = os.path.join(staging, name)
            if not os.path.exists(link_dst):
                os.symlink(full, link_dst)
            seen_bags.add(full)
            newly.append(name)
        return newly

    def stop(self):
        self._stop.set()


def register_local_incremental_gate(trip_name, trip_dir, initial_bag_groups=3):
    """Register a LocalBagGate for cached local trips (incremental parse)."""
    bag_paths = find_trip_bag_paths(trip_dir)
    slots = _build_bag_time_slots(bag_paths)
    gate = LocalBagGate(bag_paths, initial_groups=initial_bag_groups)
    gate.request_more_bags(initial_bag_groups)
    _ACTIVE_DOWNLOADERS[trip_name] = gate
    print("[info] Local incremental gate: {}/{} time slots, {} bags total, "
          "initial batch {} groups".format(
              gate._released, len(slots), gate._total_bags, initial_bag_groups))
    return gate


def stage_bags_symlink(bag_paths, staging_dir):
    """
    Create symlinks named by bag basename inside ``staging_dir``.

    Warn on duplicate basenames — keeps the first occurrence.

    Args:
        bag_paths (list): Absolute paths.
        staging_dir (str): Directory receiving symlinks.

    Returns:
        str: Absolute path of staging directory suitable for preparer consumption.
    """
    os.makedirs(staging_dir, exist_ok=True)
    seen = {}
    for src in sorted(set(os.path.abspath(p) for p in bag_paths)):
        if not os.path.isfile(src):
            raise FileNotFoundError("Missing bag file: {}".format(src))
        name = os.path.basename(src)
        if name in seen:
            alt = "{}__{}".format(os.path.splitext(name)[0], abs(hash(src)) % 100000)
            name = alt + ".bag"
            while name in seen:
                alt += "_dup"
                name = alt + ".bag"
            _LOGGER.warning("Duplicate bag basename symlinked as %s", name)
        dst = os.path.join(staging_dir, name)
        if os.path.lexists(dst):
            os.unlink(dst)
        os.symlink(src, dst)
        seen[name] = True
    return os.path.abspath(staging_dir)


# =============================================================================
# Section: KITTI calib / IO helpers
# =============================================================================
def parse_kitti_calib_txt(calib_path):
    """
    Parse ``sequences/<seq>/calib.txt`` produced by dataset preparer.

    Returns:
        dict with keys ``K`` (3x3), ``Tr`` (3x4 camera→lidar rows),
        ``T_lidar_to_cam`` (4x4 inferred from ``Tr``).
    """
    p2_vals = None
    tr_vals = None
    with io.open(calib_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line.startswith("P2:"):
                p2_vals = [float(v) for v in line.split()[1:]]
            elif line.startswith("Tr:"):
                tr_vals = [float(v) for v in line.split()[1:]]
            if p2_vals is not None and tr_vals is not None:
                break
    if p2_vals is None or tr_vals is None:
        raise ValueError("Invalid calib file (missing P2 or Tr): {}".format(calib_path))
    if len(p2_vals) != 12 or len(tr_vals) != 12:
        raise ValueError("Unexpected P2/Tr length in {}".format(calib_path))
    K = np.eye(3, dtype=np.float64)
    K[0, 0] = p2_vals[0]
    K[0, 2] = p2_vals[2]
    K[1, 1] = p2_vals[5]
    K[1, 2] = p2_vals[6]
    Tr = np.array(tr_vals, dtype=np.float64).reshape(3, 4)
    T_cl = np.eye(4, dtype=np.float64)
    T_cl[:3, :] = Tr
    T_lidar_to_cam = np.linalg.inv(T_cl)
    return {"K": K, "Tr": Tr, "T_lidar_to_cam": T_lidar_to_cam}


def read_times_file(times_path):
    """Load ``times.txt`` relative seconds (one float per line)."""
    times = []
    with io.open(times_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            times.append(float(line))
    return np.array(times, dtype=np.float64)


def read_kitti_poses(pose_path):
    """
    Load KITTI pose file (3x4 rows per frame) into list of 4x4 matrices.

    These transforms map points from frame ``i`` into frame ``0`` coordinates.
    """
    poses = []
    with io.open(pose_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            vals = [float(v) for v in line.split()]
            if len(vals) != 12:
                continue
            T = np.eye(4, dtype=np.float64)
            T[:3, :] = np.array(vals, dtype=np.float64).reshape(3, 4)
            poses.append(T)
    return poses


def list_sequence_frames(seq_dir):
    """Return sorted 6-digit frame stems available in ``image_2`` with matching velodyne."""
    image_dir = os.path.join(seq_dir, "image_2")
    velo_dir = os.path.join(seq_dir, "velodyne")
    if not os.path.isdir(image_dir) or not os.path.isdir(velo_dir):
        raise FileNotFoundError("Missing image_2/ or velodyne/ under {}".format(seq_dir))
    _IMG_EXTS = ('.png', '.jpg', '.jpeg')
    stems = []
    for name in sorted(os.listdir(image_dir)):
        if not any(name.endswith(e) for e in _IMG_EXTS):
            continue
        stem = os.path.splitext(name)[0]
        if os.path.isfile(os.path.join(velo_dir, stem + ".bin")):
            stems.append(stem)
    if not stems:
        raise RuntimeError("No synced image/bin pairs in {}".format(seq_dir))
    return stems


def filter_pointcloud_for_bev(pcd, xbound, ybound, zbound):
    """
    Mirror ``CustomDataset`` ego + range filtering for stable inference inputs.

    Args:
        pcd (np.ndarray): (N,3) or (N,4+)
        xbound,ybound,zbound: tuples (min,max,step) from bev_settings.

    Returns:
        np.ndarray: filtered (M,3+)
    """
    if pcd.size == 0:
        return pcd
    x_min, x_max = xbound[0], xbound[1]
    y_min, y_max = ybound[0], ybound[1]
    z_min, z_max = zbound[0], zbound[1]

    ego = (np.abs(pcd[:, 0]) > 3.0) | (np.abs(pcd[:, 1]) > 3.0)
    pcd = pcd[ego, :]
    if pcd.shape[0] == 0:
        return pcd

    mask = (
        (pcd[:, 0] >= x_min)
        & (pcd[:, 0] <= x_max)
        & (pcd[:, 1] >= y_min)
        & (pcd[:, 1] <= y_max)
        & (pcd[:, 2] >= z_min)
        & (pcd[:, 2] <= z_max)
    )
    pcd = pcd[mask, :]

    if pcd.shape[0] < 100:
        # fallback mild filters (mirror CustomDataset)
        raw = pcd  # noqa: intentionally tight
        if raw.size == 0:
            return raw
        ego2 = (np.abs(raw[:, 0]) > 2.0) | (np.abs(raw[:, 1]) > 2.0)
        raw = raw[ego2, :]
        mask2 = (raw[:, 0] >= x_min) & (raw[:, 0] <= x_max) & (raw[:, 1] >= y_min) & (raw[:, 1] <= y_max)
        raw = raw[mask2, :]
        return raw

    return pcd


def load_raw_pointcloud(bin_path):
    raw = np.fromfile(bin_path, dtype=np.float32)
    if raw.size % 4 == 0:
        return raw.reshape(-1, 4)
    if raw.size % 3 == 0:
        return raw.reshape(-1, 3)
    raise ValueError("Bad point cloud bin: {}".format(bin_path))


# =============================================================================
# Section: Sampling strategies
# =============================================================================
def apply_sample_strategy(indices, stems, strategy, interval, poses_path, times_path):
    """
    Subselect frame indices according to temporal / distance heuristic.

    Args:
        indices (list[int]): Enumeration positions for ``stems``.
        stems (list[str]): Sorted frame stems.
        strategy (str): ``time``, ``distance``, or ``none``.
        interval (float): Seconds (time) or meters (distance).
        poses_path (str|None): KITTI pose file path.
        times_path (str|None): Relative seconds file path.

    Returns:
        list[int]: Indices into ``indices`` preserving chronological order subset.
    """
    if strategy == "none":
        return list(range(len(indices)))

    if strategy == "adaptive":
        if poses_path is None or not os.path.isfile(poses_path):
            _LOGGER.warning("adaptive needs poses — falling back to time strategy.")
            return apply_sample_strategy(indices, stems, "time", interval, poses_path, times_path)
        poses = read_kitti_poses(poses_path)
        if len(poses) < len(stems):
            while len(poses) < len(stems):
                poses.append(poses[-1])
        sel = [0]
        for i in range(1, len(stems)):
            cur = poses[i][:3, 3]
            prev = poses[i - 1][:3, 3]
            speed = float(np.linalg.norm(cur - prev)) * 10.0
            if speed < 0.3:
                adaptive_interval = interval * 5.0
            elif speed > 20.0:
                adaptive_interval = interval * 0.5
            else:
                adaptive_interval = interval
            last_pos = poses[sel[-1]][:3, 3]
            dist_since = float(np.linalg.norm(cur - last_pos))
            if dist_since >= adaptive_interval:
                sel.append(i)
        return sel

    sel = [0]

    if strategy == "time":
        if times_path is None or not os.path.isfile(times_path):
            _LOGGER.warning("times.txt missing — falling back to uniform stride heuristic.")
            stride = max(1, int(round(interval * 10.0)))
            return indices[::stride]
        t = read_times_file(times_path)
        if len(t) != len(stems):
            _LOGGER.warning("times.txt mismatch — using index-based surrogate timing.")
            t = np.linspace(0.0, float(len(stems) - 1), num=len(stems))
        anchor_t = float(t[0])
        for i in range(1, len(stems)):
            if float(t[i]) - anchor_t >= interval:
                sel.append(i)
                anchor_t = float(t[i])
        return sel

    if strategy == "distance":
        if poses_path is None or not os.path.isfile(poses_path):
            _LOGGER.warning(
                "poses missing — falling back to time sampling with interval=%s",
                interval,
            )
            return apply_sample_strategy(indices, stems, "time", interval, None, times_path)
        poses = read_kitti_poses(poses_path)
        if len(poses) < len(stems):
            _LOGGER.warning("Short pose file — padding with last pose.")
            while len(poses) < len(stems):
                poses.append(poses[-1])
        dist_accum = 0.0
        for i in range(1, len(stems)):
            cur = poses[i][:3, 3]
            prev = poses[i - 1][:3, 3]
            dist_accum += float(np.linalg.norm(cur - prev))
            if dist_accum >= interval:
                sel.append(i)
                dist_accum = 0.0
        return sel

    raise ValueError("Unknown sample strategy: {}".format(strategy))


# =============================================================================
# Section: Model batching
# =============================================================================
def collate_and_pad_batch(img_bgr_list, pc_list, init_T, K_list, device, torch_mod):
    """
    Convert numpy lists into padded torch batch matching BEVCalibInference forward.

    Args:
        img_bgr_list (list[np.ndarray]): HxWx3 uint8 BGR already resized to model input.
        pc_list (list[np.ndarray]): Each (Ni,3) LiDAR points after filtering.
        init_T (np.ndarray): (4,4) initial LiDAR→Camera (broadcast per batch item).
        K_list (list[np.ndarray]): (3,3) intrinsics matching resized image.
        device (torch.device): Target device.
        torch_mod: imported torch module.

    Returns:
        tuple: (imgs_t, pcs_t, init_T_t, post_T_t, K_t)
    """
    assert len(img_bgr_list) == len(pc_list) == len(K_list)
    B = len(img_bgr_list)
    if B == 0:
        return None

    max_pts = max(p.shape[0] for p in pc_list)
    h, w = img_bgr_list[0].shape[:2]

    imgs = np.stack(img_bgr_list, axis=0).astype(np.float32)
    imgs_t = torch_mod.from_numpy(imgs).permute(0, 3, 1, 2).contiguous().to(device)

    pcs = np.zeros((B, max_pts, 3), dtype=np.float32)
    for i, p in enumerate(pc_list):
        n = p.shape[0]
        if n == 0:
            pcs[i, :, :] = 0.0
            continue
        pcs[i, :n, :] = p[:, :3]
        if n < max_pts:
            pcs[i, n:, :] = 999999.0
    pcs_t = torch_mod.from_numpy(pcs).to(device)

    init_np = np.repeat(init_T[None, :, :].astype(np.float32), B, axis=0)
    init_T_t = torch_mod.from_numpy(init_np).to(device)

    post_T_t = torch_mod.eye(4, device=device, dtype=torch_mod.float32).unsqueeze(0).expand(B, -1, -1).contiguous()

    K_np = np.stack(K_list, axis=0).astype(np.float32)
    K_t = torch_mod.from_numpy(K_np).to(device)

    return imgs_t, pcs_t, init_T_t, post_T_t, K_t


def run_inference_batches(
    frames,
    T_init,
    wrapper,
    device_str,
    img_shape_hw,
    batch_size,
    torch_mod,
):
    """
    Iterate prepared frames and collect per-frame predicted LiDAR→Camera transforms.

    Args:
        frames (list[dict]): Each dict with keys ``img_bgr`` (resized), ``pc`` (N,3), ``K`` (3,3).
        T_init (np.ndarray): Initial 4x4 LiDAR→Camera (config / calib derived).
        wrapper: ``BEVCalibInference`` module.
        device_str (str): e.g. ``cuda:0`` or ``cpu``.
        img_shape_hw (tuple): (H,W) model input.
        batch_size (int): Micro-batch size.

    Returns:
        list[np.ndarray]: length == len(frames), each (4,4) float32/float64.
    """
    device = torch_mod.device(device_str)
    preds = []
    with torch_mod.no_grad():
        for start in range(0, len(frames), batch_size):
            chunk = frames[start : start + batch_size]
            imgs = [f["img_bgr"] for f in chunk]
            pcs = [f["pc"] for f in chunk]
            ks = [f["K"] for f in chunk]
            batch = collate_and_pad_batch(imgs, pcs, T_init, ks, device, torch_mod)
            if batch is None:
                continue
            imgs_t, pcs_t, init_T_t, post_T_t, K_t = batch
            use_amp = device_str.startswith("cuda") and not getattr(wrapper, "disable_amp", False)
            if use_amp:
                with torch_mod.cuda.amp.autocast():
                    pred = wrapper(imgs_t, pcs_t, init_T_t, post_T_t, K_t)
            else:
                pred = wrapper(imgs_t, pcs_t, init_T_t, post_T_t, K_t)
            pred_np = pred.detach().cpu().numpy()
            for i in range(pred_np.shape[0]):
                preds.append(pred_np[i])
    return preds


def _perturb_T_rpy(T_init, rpy_deg):
    """Apply xyz Euler perturbation (degrees) to T_init rotation."""
    ScipyRot = _require_scipy_rot()
    T_out = np.array(T_init, dtype=np.float64, copy=True)
    R_delta = ScipyRot.from_euler("xyz", rpy_deg, degrees=True).as_matrix()
    T_out[:3, :3] = R_delta @ T_out[:3, :3]
    return T_out


def _build_multi_init_candidates(T_init, sweep_deg, grid=False):
    """Build T_init candidates for multi-init scan (axis-wise or full grid)."""
    sweep = float(sweep_deg)
    if sweep <= 0:
        return [("base", np.array(T_init, copy=True))]

    candidates = [("base", np.array(T_init, copy=True))]
    if grid:
        steps = [-sweep, -sweep * 0.5, 0.0, sweep * 0.5, sweep]
        for r in steps:
            for p in steps:
                for y in steps:
                    if abs(r) + abs(p) + abs(y) < 1e-9:
                        continue
                    label = "R{:+.1f}_P{:+.1f}_Y{:+.1f}".format(r, p, y)
                    candidates.append((label, _perturb_T_rpy(T_init, (r, p, y))))
        return candidates

    for axis, tag in ((0, "R"), (1, "P"), (2, "Y")):
        for sign in (+1.0, -1.0):
            rpy = [0.0, 0.0, 0.0]
            rpy[axis] = sign * sweep
            label = "{}{:+.1f}".format(tag, rpy[axis])
            candidates.append((label, _perturb_T_rpy(T_init, tuple(rpy))))
    return candidates


def _prepare_frames_for_reinfer(frame_meta, cv2, xbound, ybound, zbound, Wm, Hm):
    """Load resized BGR + filtered PC + K from accepted frame meta."""
    frames = []
    for meta in frame_meta:
        img_path = meta.get("img_path")
        pc_path = meta.get("pc_path")
        K = meta.get("K")
        if not img_path or not pc_path or K is None:
            continue
        im = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if im is None:
            continue
        ih, iw = im.shape[:2]
        sx = float(Wm) / float(iw)
        sy = float(Hm) / float(ih)
        Ks = np.array(K, copy=True)
        Ks[0, 0] *= sx
        Ks[0, 2] *= sx
        Ks[1, 1] *= sy
        Ks[1, 2] *= sy
        im_r = cv2.resize(im, (Wm, Hm), interpolation=cv2.INTER_AREA)
        pc_raw = load_raw_pointcloud(pc_path)
        pc_f = filter_pointcloud_for_bev(pc_raw, xbound, ybound, zbound)
        if pc_f.shape[0] < 10:
            continue
        frames.append({
            "img_bgr": im_r,
            "pc": pc_f[:, :3].astype(np.float32),
            "K": Ks,
            "meta": meta,
        })
    return frames


def _aggregate_predictions(predictions, max_frames, min_agg_frames):
    """Temporal aggregate with optional uniform subsample to max_frames."""
    aggregator = TemporalCalibrationAggregator(
        min_frames=min_agg_frames, max_frames=max_frames, method="axis_angle_median",
    )
    n = len(predictions)
    if n > max_frames:
        sel = sorted(set(int(round(i)) for i in np.linspace(0, n - 1, num=max_frames)))
        for idx in sel:
            aggregator.add(predictions[idx])
    else:
        for p in predictions:
            aggregator.add(p)
    return aggregator.aggregate(), aggregator.get_confidence(), aggregator.count


def _multi_init_select_best(
    T_init_base,
    frame_meta,
    base_predictions,
    sweep_deg,
    grid,
    wrapper,
    device,
    batch_size,
    img_shape,
    max_frames,
    min_agg_frames,
    T_gt_ref,
    cv2,
    torch_mod,
    xbound,
    ybound,
    zbound,
):
    """
    Re-infer with perturbed T_init candidates; pick best by GT residual or confidence.

    Returns:
        (best_T_cal, best_T_init_used, best_conf, scan_rows)
    """
    Hm, Wm = int(img_shape[0]), int(img_shape[1])
    frames = _prepare_frames_for_reinfer(
        frame_meta, cv2, xbound, ybound, zbound, Wm, Hm)
    if not frames:
        raise RuntimeError("multi_init: no frames could be loaded for re-inference")

    n_use = min(len(base_predictions), len(frames))
    frame_meta = frame_meta[:n_use]
    frames = frames[:n_use]
    base_predictions = base_predictions[:n_use]

    candidates = _build_multi_init_candidates(T_init_base, sweep_deg, grid=grid)
    scan_rows = []
    best = None

    for label, T_cand in candidates:
        if label == "base":
            preds = base_predictions
        else:
            preds = run_inference_batches(
                frames, T_cand, wrapper, device, img_shape, batch_size, torch_mod)
        if len(preds) != n_use:
            continue
        T_cal, conf, n_acc = _aggregate_predictions(preds, max_frames, min_agg_frames)
        row = {
            "label": label,
            "T_init": T_cand,
            "T_cal": T_cal,
            "conf": conf,
            "n_acc": n_acc,
            "init_cal_deg": T_to_calibration_metrics(T_cand, T_cal)["rot_geodesic_deg"],
        }
        if T_gt_ref is not None:
            bm = _compute_bias_compensation_metrics(T_cand, T_cal, T_gt_ref)
            row["residual_deg"] = bm["residual_bias_deg"]
            row["recover_pct"] = bm["compensation_ratio_pct"]
            row["shortcut"] = bm["shortcut_risk"]
            score = row["residual_deg"]
        else:
            row["residual_deg"] = float("nan")
            row["recover_pct"] = float("nan")
            row["shortcut"] = ""
            _cpct = conf.get("confidence_pct", 0) if conf else 0
            score = -float(_cpct if _cpct is not None else 0) + row["init_cal_deg"] * 0.1
        row["score"] = score
        scan_rows.append(row)
        if best is None or score < best["score"]:
            best = row

    print("[multi_init] sweep={}° grid={} candidates={} frames={}".format(
        sweep_deg, grid, len(scan_rows), n_use))
    for row in sorted(scan_rows, key=lambda r: r["score"]):
        extra = ""
        if T_gt_ref is not None:
            extra = " residual={:.3f}° recover={:.0f}% {}".format(
                row["residual_deg"], row["recover_pct"] if row["recover_pct"] == row["recover_pct"] else 0.0,
                row["shortcut"])
        print("[multi_init]   {} init->cal={:.3f}° score={:.4f}{}".format(
            row["label"], row["init_cal_deg"], row["score"], extra))
    print("[multi_init] selected: {} (residual={:.4f}°)".format(
        best["label"], best.get("residual_deg", float("nan"))))
    return best["T_cal"], best["T_init"], best["conf"], scan_rows


# =============================================================================
# Section: Projection comparison rendering
# =============================================================================
def render_projection_comparison(
    image_bgr,
    points_xyz,
    T_original,
    T_calibrated,
    K,
    project_fn,
    render_fn,
    compute_err_fn,
    frame_label="",
    banner_extra="",
):
    """
    Build side-by-side projection comparison (original vs calibrated extrinsic).

    Args:
        image_bgr (np.ndarray): HxWx3 BGR uint8 (resized to model input).
        points_xyz (np.ndarray): (N,3) LiDAR points in LiDAR frame.
        T_original,T_calibrated (np.ndarray): LiDAR->Camera 4x4.
        K (np.ndarray): 3x3 intrinsics matching ``image_bgr`` resolution.
        project_fn, render_fn: visualization helpers.
        compute_err_fn: optional pose error dict between calibrated vs original.
        frame_label (str): Frame identification text (stem + timestamp).

    Returns:
        np.ndarray: Hx(2W)x3 BGR uint8 mosaic.
    """
    h, w = image_bgr.shape[:2]
    pts2d_o, dep_o, _ = project_fn(points_xyz, T_original, K, (h, w))
    panel_o = render_fn(image_bgr.copy(), pts2d_o, dep_o)

    pts2d_c, dep_c, _ = project_fn(points_xyz, T_calibrated, K, (h, w))
    panel_c = render_fn(image_bgr.copy(), pts2d_c, dep_c)

    mosaic = np.hstack([panel_o, panel_c])

    banner_h = 40
    banner = np.zeros((banner_h, mosaic.shape[1], 3), dtype=np.uint8)

    cv2 = _require_cv2()
    txt = "Init(L) vs Calibrated(R)"
    if banner_extra:
        txt += "  |  {}".format(banner_extra)
    if frame_label:
        txt += "  |  {}".format(frame_label)
    cv2.putText(banner, txt, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    return np.vstack([banner, mosaic])


# =============================================================================
# Section: Matrix persistence
# =============================================================================
def _project_pts_simple(points, T, K, img_hw):
    """Minimal projection without visualization.py dependency."""
    pts_h = np.hstack([points[:, :3], np.ones((points.shape[0], 1))])
    pts_cam = (T @ pts_h.T).T
    depths = pts_cam[:, 2]
    valid = depths > 0.1
    pts_cam = pts_cam[valid]
    depths = depths[valid]
    pts_2d_h = (K @ pts_cam[:, :3].T).T
    pts_2d = pts_2d_h[:, :2] / pts_2d_h[:, 2:3]
    h, w = img_hw
    in_img = (pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < w) & (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < h)
    return pts_2d[in_img], depths[in_img]


def _render_pts_on_image(image_bgr, pts_2d, depths, max_depth=80.0):
    """Depth-colored point rendering with jet colormap."""
    cv2 = _require_cv2()
    canvas = image_bgr.copy()
    if len(pts_2d) == 0:
        return canvas
    norm_d = np.clip(depths / max_depth, 0.0, 1.0)
    import matplotlib.cm as cm
    colors = (cm.jet(norm_d)[:, :3] * 255).astype(np.uint8)
    colors_bgr = colors[:, ::-1]
    for i in range(len(pts_2d)):
        x, y = int(pts_2d[i, 0]), int(pts_2d[i, 1])
        c = tuple(int(v) for v in colors_bgr[i])
        cv2.circle(canvas, (x, y), 2, c, -1)
    return canvas


def _parse_rpy_triplet(text):
    """Parse ``roll,pitch,yaw`` degrees from CLI string."""
    if not text:
        return None
    parts = [p.strip() for p in str(text).split(",")]
    if len(parts) != 3:
        raise ValueError(
            "inject_lidar_rpy_deg expects 'roll,pitch,yaw' in degrees, got: {}".format(text))
    return tuple(float(p) for p in parts)


def _apply_lidar_rpy_offset(lidar_config, rpy_deg):
    """Apply extra sensor_to_lidar rotation (degrees, xyz Euler) on top of config."""
    ScipyRot = _require_scipy_rot()
    out = dict(lidar_config)
    R_extra = ScipyRot.from_euler("xyz", rpy_deg, degrees=True).as_matrix()
    R_base = ScipyRot.from_quat(lidar_config["orientation"]).as_matrix()
    out["orientation"] = ScipyRot.from_matrix(R_extra @ R_base).as_quat()
    return out


def _resolve_gt_lidars_cfg_path(gt_lidars_cfg, config_dir):
    """Resolve explicit GT lidars cfg or auto-detect ``lidars_bk.cfg`` in config_dir."""
    if gt_lidars_cfg:
        path = os.path.abspath(gt_lidars_cfg)
        if not os.path.isfile(path):
            raise FileNotFoundError("gt_lidars_cfg not found: {}".format(path))
        return path
    if config_dir:
        auto = os.path.join(config_dir, "lidars_bk.cfg")
        if os.path.isfile(auto):
            return auto
    return None


def _load_gt_T_from_lidars_cfg(gt_lidars_cfg_path, cam_cfg):
    """Build T_lidar_to_cam ground truth from GT lidars cfg + camera cfg."""
    gt_lidars = ConfigParser.parse_lidars_cfg(gt_lidars_cfg_path)
    return compute_T_lidar_to_cam(cam_cfg, gt_lidars)


def _compute_bias_compensation_metrics(T_init, T_cal, T_gt):
    """
    Quantify systematic-bias injection vs recovery relative to true GT extrinsic.

    Returns dict used for shortcut / compensation reporting.
    """
    m_init_gt = T_to_calibration_metrics(T_gt, T_init)
    m_cal_gt = T_to_calibration_metrics(T_gt, T_cal)
    m_init_cal = T_to_calibration_metrics(T_init, T_cal)

    injected = float(m_init_gt["rot_geodesic_deg"])
    residual = float(m_cal_gt["rot_geodesic_deg"])
    improvement = injected - residual
    ratio = (improvement / injected * 100.0) if injected > 1e-6 else float("nan")
    moved_toward_gt = improvement > 0.01

    # Shortcut: tiny init->cal change while large injected bias, or poor recovery.
    init_cal = float(m_init_cal["rot_geodesic_deg"])
    if injected >= 0.5 and init_cal < 0.3 and ratio < 30.0:
        shortcut_risk = "HIGH"
        shortcut_note = (
            "Model barely moved init ({:.3f} deg) despite {:.3f} deg injected bias — "
            "likely init-locked / shortcut behaviour".format(init_cal, injected))
    elif injected >= 0.5 and ratio < 30.0:
        shortcut_risk = "HIGH"
        shortcut_note = (
            "Poor bias recovery: {:.1f}% compensated ({:.3f}/{:.3f} deg)".format(
                ratio if ratio == ratio else 0.0, improvement, injected))
    elif injected >= 0.5 and ratio < 60.0:
        shortcut_risk = "MEDIUM"
        shortcut_note = "Partial recovery ({:.1f}% of {:.3f} deg bias)".format(
            ratio if ratio == ratio else 0.0, injected)
    elif injected >= 0.5 and moved_toward_gt:
        shortcut_risk = "LOW"
        shortcut_note = "Good bias recovery ({:.1f}% of {:.3f} deg)".format(
            ratio if ratio == ratio else 0.0, injected)
    else:
        shortcut_risk = "N/A"
        shortcut_note = "Injected bias {:.3f} deg — increase --inject_lidar_rpy_deg for stress test".format(
            injected)

    return {
        "injected_bias_deg": injected,
        "residual_bias_deg": residual,
        "compensation_deg": improvement,
        "compensation_ratio_pct": ratio,
        "init_cal_delta_deg": init_cal,
        "moved_toward_gt": moved_toward_gt,
        "shortcut_risk": shortcut_risk,
        "shortcut_note": shortcut_note,
        "init_vs_gt_roll_deg": m_init_gt["roll_delta_deg"],
        "init_vs_gt_pitch_deg": m_init_gt["pitch_delta_deg"],
        "init_vs_gt_yaw_deg": m_init_gt["yaw_delta_deg"],
        "cal_vs_gt_roll_deg": m_cal_gt["roll_delta_deg"],
        "cal_vs_gt_pitch_deg": m_cal_gt["pitch_delta_deg"],
        "cal_vs_gt_yaw_deg": m_cal_gt["yaw_delta_deg"],
    }


def _check_install_angle_error_is_gt(cam_iae, lidar_iae, threshold=2.5):
    """
    Determine if config install_angle_error values indicate GT-quality calibration.

    Both camera and lidar must have install_angle_error fields with all xyz
    absolute values below ``threshold`` degrees.

    Returns:
        tuple: (is_gt: bool, reason: str)
    """
    if cam_iae is None:
        return False, "traffic_2 camera has no install_angle_error field"
    if lidar_iae is None:
        return False, "main LiDAR has no install_angle_error field"

    cam_vals = [abs(cam_iae.get("x", 999)), abs(cam_iae.get("y", 999)), abs(cam_iae.get("z", 999))]
    lid_vals = [abs(lidar_iae.get("x", 999)), abs(lidar_iae.get("y", 999)), abs(lidar_iae.get("z", 999))]

    for axis, v in zip(["x", "y", "z"], cam_vals):
        if v >= threshold:
            return False, "traffic_2 install_angle_error.{} = {:.3f} deg (>= {})".format(axis, v, threshold)
    for axis, v in zip(["x", "y", "z"], lid_vals):
        if v >= threshold:
            return False, "main LiDAR install_angle_error.{} = {:.3f} deg (>= {})".format(axis, v, threshold)

    return True, "both traffic_2 and main LiDAR install_angle_error within +/-{} deg".format(threshold)


def _compute_multi_window_errors(all_preds, T_gt, window_sizes=(50, 100, 200, 400)):
    """
    Compute aggregation error at multiple frame windows via uniform subsampling.

    For each window size N, uniformly sample N predictions from the full set,
    aggregate via axis-angle median, compare to GT.

    Args:
        all_preds (list): List of prediction matrices (4x4 numpy arrays).
        T_gt (np.ndarray): Ground truth 4x4 transform for comparison.
        window_sizes (tuple): Frame counts to evaluate.

    Returns:
        dict: {window_size: {"rot": float, "roll": float, "pitch": float, "yaw": float}}
    """
    ScipyRot = _require_scipy_rot()
    if not all_preds:
        return {}

    n_total = len(all_preds)
    results = {}

    for ws in window_sizes:
        if ws > n_total:
            continue
        indices = np.linspace(0, n_total - 1, num=ws, dtype=int)
        sampled = [all_preds[i] for i in indices]

        tmp_agg = TemporalCalibrationAggregator(
            min_frames=1, max_frames=ws, method="axis_angle_median",
        )
        for pred in sampled:
            tmp_agg.add(pred)
        T_agg = tmp_agg.aggregate()

        metrics = T_to_calibration_metrics(T_gt, T_agg)
        results[ws] = {
            "rot": metrics["rot_geodesic_deg"],
            "roll": abs(metrics.get("roll_delta_deg", 0.0)),
            "pitch": abs(metrics.get("pitch_delta_deg", 0.0)),
            "yaw": abs(metrics.get("yaw_delta_deg", 0.0)),
        }

    return results


def _try_load_gt_extrinsic(trip_dir, trip_out):
    """
    Try to find a ground-truth extrinsic matrix for comparison.

    Checks: ``<trip_dir>/model/lidars.cfg`` + ``cameras.cfg`` (factory-calibrated),
    or ``<trip_out>/gt_extrinsic.txt`` (user-provided).
    """
    if trip_out:
        gt_path = os.path.join(trip_out, "gt_extrinsic.txt")
        if os.path.isfile(gt_path):
            try:
                T = np.loadtxt(gt_path)
                if T.shape == (4, 4):
                    return T
            except Exception:
                pass
    if trip_dir:
        model_cam = os.path.join(trip_dir, "model", "cameras.cfg")
        model_lid = os.path.join(trip_dir, "model", "lidars.cfg")
        if os.path.isfile(model_cam) and os.path.isfile(model_lid):
            try:
                cameras = ConfigParser.parse_cameras_cfg(model_cam)
                lidars = ConfigParser.parse_lidars_cfg(model_lid)
                cam_name = "traffic_2"
                if cam_name in cameras:
                    return compute_T_lidar_to_cam(cameras[cam_name], lidars)
            except Exception:
                pass
    return None


def _fallback_projection(image_bgr, points, T_orig, T_cal, K, cv2, frame_label=""):
    """Side-by-side projection comparison without external visualization module."""
    h, w = image_bgr.shape[:2]
    pts2d_o, dep_o = _project_pts_simple(points, T_orig, K, (h, w))
    panel_o = _render_pts_on_image(image_bgr.copy(), pts2d_o, dep_o)
    pts2d_c, dep_c = _project_pts_simple(points, T_cal, K, (h, w))
    panel_c = _render_pts_on_image(image_bgr.copy(), pts2d_c, dep_c)

    mosaic = np.hstack([panel_o, panel_c])
    banner_h = 40
    banner = np.zeros((banner_h, mosaic.shape[1], 3), dtype=np.uint8)
    txt = "Original(L) vs Calibrated(R)"
    if frame_label:
        txt += "  |  {}".format(frame_label)
    cv2.putText(banner, txt, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([banner, mosaic])


def save_matrix_txt(path, T):
    with io.open(path, "w", encoding="utf-8") as handle:
        for row in T:
            handle.write(" ".join("{:.12e}".format(v) for v in row))
            handle.write("\n")


# =============================================================================
# Section: Aggregate reporting
# =============================================================================
def summarize_confidence(conf_dict):
    """Render TemporalCalibrationAggregator confidence dict into text."""
    if not conf_dict:
        return "(no confidence — insufficient frames)"
    parts = []
    def _nan_safe(v):
        if v is None:
            return float("nan")
        return v
    parts.append(
        "roll_std={:.4f} deg  pitch_std={:.4f} deg  yaw_std={:.4f} deg".format(
            _nan_safe(conf_dict.get("roll_std")),
            _nan_safe(conf_dict.get("pitch_std")),
            _nan_safe(conf_dict.get("yaw_std")),
        )
    )
    parts.append("total_std={:.4f} deg  n_frames={}".format(
        _nan_safe(conf_dict.get("total_std")),
        conf_dict.get("n_frames", -1),
    ))
    return "\n".join(parts)


def _compute_confidence_pct(r):
    """Compute a 0-100% confidence score from frame count and prediction std.

    Score = frame_factor * stability_factor * 100
      - frame_factor: min(1.0, n_frames / 200) — saturates at 200 frames
      - stability_factor: max(0, 1 - total_std / 0.15) — 0.15 deg = zero confidence
    """
    n = r.get("frames") or 0
    std = r.get("total_std")
    if std is None or not isinstance(std, (int, float)) or (isinstance(std, float) and std != std):
        std = 0.15
    frame_f = min(1.0, n / 200.0) if n > 0 else 0
    stab_f = max(0.0, 1.0 - std / 0.15)
    return round(frame_f * stab_f * 100, 1)


def _detect_anomalous_trips(ok_rows):
    """Detect trips with anomalous calibration metrics and generate diagnostic text.

    An anomaly is flagged when a trip's metric deviates > 2 sigma from the group mean,
    or exceeds absolute thresholds.
    """
    if len(ok_rows) < 2:
        return []

    def _best_medw(r):
        raw = r.get("multi_window_errors", {})
        mwe = {int(k): v for k, v in raw.items()} if raw else {}
        for ws in (50, 100, 200):
            if ws in mwe:
                return mwe[ws]
        return None

    medw_vals = []
    roll_vals = []
    pitch_vals = []
    yaw_vals = []
    for r in ok_rows:
        e = _best_medw(r)
        if e:
            def _ev(k, default=0):
                v = e.get(k)
                return float(v) if v is not None else default
            medw_vals.append(_ev("rot", float("nan")))
            roll_vals.append(abs(_ev("roll", 0)))
            pitch_vals.append(abs(_ev("pitch", 0)))
            yaw_vals.append(abs(_ev("yaw", 0)))

    if len(medw_vals) < 2:
        return []

    def _stats(vals):
        m = float(np.mean(vals))
        s = float(np.std(vals))
        return m, s

    medw_mean, medw_std = _stats(medw_vals)
    roll_mean, roll_std = _stats(roll_vals)
    pitch_mean, pitch_std = _stats(pitch_vals)
    yaw_mean, yaw_std = _stats(yaw_vals)

    anomalies = []
    for i, r in enumerate(ok_rows):
        trip = r.get("trip", "")
        e = _best_medw(r)
        if not e:
            continue
        rot = e.get("rot", 0)
        diags = []

        _outlier_thresh = 1.5 if len(ok_rows) <= 5 else 2.0
        if medw_std > 0 and rot > medw_mean and (rot - medw_mean) > _outlier_thresh * medw_std:
            diags.append("Total MEDW {:.4f} deg deviates {:.1f} sigma from mean {:.4f} deg".format(
                rot, abs(rot - medw_mean) / medw_std, medw_mean))

        for axis, val, a_mean, a_std in [
            ("Roll", abs(e.get("roll", 0)), roll_mean, roll_std),
            ("Pitch", abs(e.get("pitch", 0)), pitch_mean, pitch_std),
            ("Yaw", abs(e.get("yaw", 0)), yaw_mean, yaw_std),
        ]:
            if a_std > 0 and val > a_mean and abs(val - a_mean) > _outlier_thresh * a_std:
                diags.append("{} error {:.4f} deg deviates {:.1f} sigma from mean {:.4f} deg".format(
                    axis, val, abs(val - a_mean) / a_std, a_mean))

        if not diags:
            continue

        orig_iae = r.get("orig_iae", {})
        iae_total = math.sqrt(sum(v**2 for v in orig_iae.values())) if orig_iae else 0
        conf_pct = _compute_confidence_pct(r)

        detail_parts = []
        detail_parts.append("- **Anomaly**: " + "; ".join(diags))
        _ts_val = r.get("total_std")
        _ts_display = float(_ts_val) if _ts_val is not None else 0.0
        detail_parts.append("- **Confidence**: {:.0f}% (total_std={:.4f} deg)".format(
            conf_pct, _ts_display))
        detail_parts.append("- **Original IAE magnitude**: {:.4f} deg (roll={:.3f}, pitch={:.3f}, yaw={:.3f})".format(
            iae_total, orig_iae.get("roll", 0), orig_iae.get("pitch", 0), orig_iae.get("yaw", 0)))

        if iae_total > 1.5:
            detail_parts.append("- **Likely cause**: Large install angle error ({:.2f} deg) — the LiDAR has significant "
                                "mounting deviation from factory standard. The model may have reduced accuracy at "
                                "extreme IAE ranges. Consider verifying the initial extrinsic.".format(iae_total))
        elif conf_pct < 80:
            detail_parts.append("- **Likely cause**: Low confidence ({:.0f}%) — insufficient frames or high prediction "
                                "variance. Consider collecting more data for this trip.".format(conf_pct))
        else:
            detail_parts.append("- **Likely cause**: Vehicle-specific characteristic or unusual driving conditions. "
                                "Inspect projection images for visual verification.")

        tag = "Elevated MEDW" if rot > medw_mean + medw_std else "Axis deviation"
        anomalies.append({"trip": trip, "tag": tag, "detail": "\n".join(detail_parts)})

    return anomalies


def build_markdown_summary(rows, global_stats):
    """Build Feishu-friendly Markdown summary."""
    lines = []
    lines.append("# BEVCalib Multi-Trip Calibration Summary")
    lines.append("")
    lines.append("## Glossary")
    lines.append("")
    lines.append("| Variable | Description |")
    lines.append("| --- | --- |")
    lines.append("| Frames | Number of frames that passed quality filters and were used for inference |")
    lines.append("| Conf% | Confidence = `min(1, frames/200) * max(0, 1 - total_std/0.15) * 100`. Higher = more reliable |")
    lines.append("| Calib Delta (deg) | Geodesic rotation angle between original and calibrated extrinsic |")
    lines.append("| Roll/Pitch/Yaw Delta | Per-axis rotation gap between original and calibrated extrinsic |")
    lines.append("| Orig IAE Roll/Pitch/Yaw | Install Angle Error of the **original trip** extrinsic vs factory standard |")
    lines.append("| Cal IAE Roll/Pitch/Yaw | Install Angle Error of the **calibrated** extrinsic vs factory standard |")
    lines.append("| total_std (deg) | L2 norm of per-axis RPY prediction std across inferred frames. Lower = more consistent |")
    lines.append("| MEDW (e.g. MEDW50) | Mean Error over Decaying Window vs reference extrinsic (true GT with --gt_lidars_cfg) |")
    lines.append("| Injected / Residual / Compensation | Systematic bias vs true GT before/after calibration; ratio = recovery % |")
    lines.append("| Shortcut | HIGH = model stays near init despite large injected bias (init-locked shortcut) |")
    lines.append("| Cross-Vehicle Consistency | Groups trips by vehicle platform, computes MEDW mean/std per group, then overall CV (coefficient of variation). CV < 10% = strong generalization |")
    lines.append("| IAE Improvement | Compares total IAE magnitude (L2 of RPY) before and after calibration. IMPROVED = closer to factory; SHIFTED = farther |")
    lines.append("")
    lines.append("## Per-trip results")
    lines.append("")

    def _fmt_iae_compact(iae_dict):
        if not iae_dict:
            return "-"
        def _v(k):
            val = iae_dict.get(k, 0)
            return float(val) if val is not None else 0.0
        return "{:.3f}/{:.3f}/{:.3f}".format(_v("roll"), _v("pitch"), _v("yaw"))

    header_cols = [
        "Trip", "Status", "Frames", "Conf%",
        "Calib Δ (deg)", "Injected", "Residual", "Recover%",
        "Shortcut", "Roll Δ", "Pitch Δ", "Yaw Δ",
        "Orig IAE (R/P/Y)", "Cal IAE (R/P/Y)",
        "total_std", "Time (s)",
    ]
    lines.append("| " + " | ".join(header_cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(header_cols)) + " |")

    for r in rows:
        status_str = r.get("status", "")
        conf_pct = _compute_confidence_pct(r) if status_str == "ok" else 0
        orig_iae = r.get("orig_iae", {})
        cal_iae = r.get("iae", {})
        def _safe_fmt(val, fmt="{:.4f}"):
            if val is None:
                return "-"
            try:
                return fmt.format(float(val))
            except (ValueError, TypeError):
                return str(val)

        bc = r.get("bias_compensation") or {}
        rec = bc.get("compensation_ratio_pct", r.get("compensation_ratio_pct"))
        rec_s = "{:.0f}%".format(rec) if rec is not None and rec == rec else "-"

        cols = [
            r.get("trip", ""),
            status_str,
            str(r.get("frames", "")),
            "{:.0f}%".format(conf_pct),
            _safe_fmt(r.get("rot_delta")),
            _safe_fmt(r.get("injected_bias_deg", bc.get("injected_bias_deg"))),
            _safe_fmt(r.get("residual_bias_deg", bc.get("residual_bias_deg"))),
            rec_s,
            str(r.get("shortcut_risk", bc.get("shortcut_risk", "-"))),
            _safe_fmt(r.get("roll_delta")),
            _safe_fmt(r.get("pitch_delta")),
            _safe_fmt(r.get("yaw_delta")),
            _fmt_iae_compact(orig_iae),
            _fmt_iae_compact(cal_iae),
            _safe_fmt(r.get("total_std")),
            _safe_fmt(r.get("total_sec", 0), "{:.1f}"),
        ]
        lines.append("| " + " | ".join(cols) + " |")

    ok_rows = [r for r in rows if r.get("status") == "ok"]
    if ok_rows:
        lines.append("")
        lines.append("## Calibration Accuracy (MEDW with RPY breakdown)")
        lines.append("")
        def _fmt_medw_cell(e):
            if not e:
                return "-"
            def _nv(v):
                if v is None:
                    return float("nan")
                return float(v)
            rot = _nv(e.get("rot"))
            r_ = _nv(e.get("roll"))
            p_ = _nv(e.get("pitch"))
            y_ = _nv(e.get("yaw"))
            if rot != rot:
                return "-"
            return "{:.4f} (R{:.4f} P{:.4f} Y{:.4f})".format(rot, r_, p_, y_)

        lines.append("| Trip | Conf% | MEDW50 | MEDW100 | MEDW200 |")
        lines.append("| --- | ---: | --- | --- | --- |")
        for r in ok_rows:
            raw_mwe = r.get("multi_window_errors", {})
            mwe = {int(k): v for k, v in raw_mwe.items()} if raw_mwe else {}
            conf_pct = _compute_confidence_pct(r)
            cols = [
                r.get("trip", ""),
                "{:.0f}%".format(conf_pct),
                _fmt_medw_cell(mwe.get(50)),
                _fmt_medw_cell(mwe.get(100)),
                _fmt_medw_cell(mwe.get(200)),
            ]
            lines.append("| " + " | ".join(cols) + " |")
    lines.append("")

    if len(ok_rows) >= 2:
        lines.append("## Model Generalization Analysis")
        lines.append("")

        vehicle_groups = {}
        for r in ok_rows:
            trip = r.get("trip", "")
            parts = trip.split("-")
            vtype = parts[1] if len(parts) >= 3 else trip.split("_")[0]
            vehicle_groups.setdefault(vtype, []).append(r)

        lines.append("### Cross-Vehicle Consistency")
        lines.append("")
        lines.append("| Vehicle Type | Trips | Mean MEDW (deg) | Std MEDW | Mean Rot Δ | Std Rot Δ | Verdict |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | --- |")

        all_medw = []
        for vtype, vrows in sorted(vehicle_groups.items()):
            medws = []
            rots = []
            for r in vrows:
                raw_mwe = r.get("multi_window_errors", {})
                mwe = {int(k): v for k, v in raw_mwe.items()} if raw_mwe else {}
                for ws in (50, 100, 200):
                    if ws in mwe and mwe[ws]:
                        _rv = mwe[ws].get("rot")
                        medws.append(float(_rv) if _rv is not None else float("nan"))
                        break
                rd = r.get("rot_delta")
                if rd is None:
                    rd = float("nan")
                if rd == rd:
                    rots.append(rd)

            mean_medw = float(np.mean(medws)) if medws else float("nan")
            std_medw = float(np.std(medws)) if len(medws) > 1 else 0.0
            mean_rot = float(np.mean(rots)) if rots else float("nan")
            std_rot = float(np.std(rots)) if len(rots) > 1 else 0.0
            all_medw.extend(medws)

            if std_medw < 0.005 and mean_medw < 0.05:
                verdict = "EXCELLENT"
            elif std_medw < 0.01:
                verdict = "GOOD"
            else:
                verdict = "FAIR"

            lines.append("| {} | {} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {} |".format(
                vtype, len(vrows), mean_medw, std_medw, mean_rot, std_rot, verdict))

        if len(all_medw) >= 2:
            overall_mean = float(np.mean(all_medw))
            overall_std = float(np.std(all_medw))
            cv = overall_std / overall_mean if overall_mean > 0 else float("nan")
            lines.append("")
            lines.append("**Overall**: Mean MEDW={:.4f} deg, Std={:.4f} deg, CV={:.2f}%".format(
                overall_mean, overall_std, cv * 100))
            if cv < 0.1:
                lines.append("  -> Model shows **strong cross-vehicle generalization** (CV < 10%)")
            elif cv < 0.2:
                lines.append("  -> Model shows **good generalization** (CV < 20%)")

        lines.append("")
        lines.append("### IAE Improvement Summary")
        lines.append("")
        lines.append("| Trip | Orig IAE (deg) | Cal IAE (deg) | Improvement | Direction | Model |")
        lines.append("| --- | ---: | ---: | ---: | --- | --- |")
        _iae_dirs = {"IMPROVED": 0, "STABLE": 0, "SHIFTED": 0}
        _no_model_trips = []
        for r in ok_rows:
            orig = r.get("orig_iae", {})
            cal = r.get("iae", {})
            if not orig or not cal:
                continue
            orig_total = math.sqrt(
                orig.get("roll", 0)**2 + orig.get("pitch", 0)**2 + orig.get("yaw", 0)**2)
            cal_total = math.sqrt(
                cal.get("roll", 0)**2 + cal.get("pitch", 0)**2 + cal.get("yaw", 0)**2)
            improvement = orig_total - cal_total
            pct = (improvement / orig_total * 100) if orig_total > 0 else 0
            has_model = r.get("has_model", True)
            if not has_model and abs(improvement) < 1e-6:
                direction = "NO_MODEL"
            else:
                direction = "IMPROVED" if improvement > 0.001 else (
                    "STABLE" if abs(improvement) < 0.01 else "SHIFTED")
            _iae_dirs[direction] = _iae_dirs.get(direction, 0) + 1
            model_flag = "YES" if has_model else "NO (fallback)"
            if not has_model:
                _no_model_trips.append(r.get("trip", ""))
            lines.append("| {} | {:.4f} | {:.4f} | {:+.4f} ({:+.1f}%) | {} | {} |".format(
                r.get("trip", ""), orig_total, cal_total, improvement, pct, direction, model_flag))
        lines.append("")
        total_trips = sum(_iae_dirs.values())
        parts = []
        for d in ("IMPROVED", "STABLE", "SHIFTED", "NO_MODEL"):
            cnt = _iae_dirs.get(d, 0)
            if cnt > 0:
                parts.append("{} {}".format(cnt, d))
        lines.append("**Summary**: {} out of {} trips".format(", ".join(parts), total_trips))
        if _iae_dirs.get("SHIFTED", 0) > 0:
            lines.append("")
            lines.append("> Note: SHIFTED trips have calibrated IAE larger than original. "
                         "This usually means the calibration adjustment moved slightly away from the factory standard, "
                         "but the calibration itself (MEDW) is still accurate. The factory standard may not be the true optimum.")
        if _no_model_trips:
            lines.append("")
            lines.append("> Warning: {} trip(s) missing complete model/ directory (need lidars.cfg + cameras.cfg): {}. "
                         "IAE values were computed using fallback from original config install_angle_error, "
                         "making Orig IAE and Cal IAE less reliable. Ensure model/ is downloaded from remote for accurate IAE comparison.".format(
                             len(_no_model_trips), ", ".join(_no_model_trips)))
        lines.append("")

    lines.append("## Global statistics")
    lines.append("")
    for k, v in sorted(global_stats.items()):
        if isinstance(v, float):
            fmt = "{:.6f}" if abs(v) < 1 else "{:.1f}"
            lines.append("- **{0}**: {1}".format(k, fmt.format(v)))
        else:
            lines.append("- **{0}**: {1}".format(k, v))
    lines.append("")
    if len(ok_rows) >= 2:
        anomalies = _detect_anomalous_trips(ok_rows)
        if anomalies:
            lines.append("## Anomaly Analysis")
            lines.append("")
            for a in anomalies:
                lines.append("### {} — {}".format(a["trip"], a["tag"]))
                lines.append("")
                lines.append(a["detail"])
                lines.append("")

    lines.append("## Recommendations")
    lines.append("")
    lines.append(
        "1. Trips with **total_std** > 0.15 deg often indicate scene degeneracy, rain/fog, or bad time sync — "
        "inspect ``projections/`` folders."
    )
    lines.append(
        "2. Large **Rot Δ** with low confidence std may mean the initial config was far from the operating point — "
        "verify ``configs/lidars.cfg`` main LiDAR selection."
    )
    lines.append(
        "3. If many trips fail extraction, confirm ``/localization/pose`` exists in non-Heavy bags and that "
        "image topic names match the selected ``--camera_name``."
    )
    lines.append("")
    return "\n".join(lines)


def generate_summary_rpy_charts(results, output_dir):
    """
    Generate RPY comparison bar charts across all trips and save as PNG.

    Returns list of generated chart filenames.
    """
    import matplotlib.pyplot as plt

    ok_results = [r for r in results if r.get("status") == "ok"]
    if len(ok_results) < 1:
        return []

    trip_names = [r.get("trip", "?") for r in ok_results]
    short_names = [n[:20] if len(n) > 20 else n for n in trip_names]
    def _f(v):
        return v if isinstance(v, (int, float)) else 0.0

    rolls = [_f(r.get("roll_delta")) for r in ok_results]
    pitches = [_f(r.get("pitch_delta")) for r in ok_results]
    yaws = [_f(r.get("yaw_delta")) for r in ok_results]
    totals = [_f(r.get("rot_delta")) for r in ok_results]
    roll_stds = [_f(r.get("roll_std")) for r in ok_results]
    pitch_stds = [_f(r.get("pitch_std")) for r in ok_results]
    yaw_stds = [_f(r.get("yaw_std")) for r in ok_results]

    charts = []

    x = np.arange(len(ok_results))
    width = 0.22

    fig, ax = plt.subplots(figsize=(max(8, len(ok_results) * 1.5), 5))
    bars_r = ax.bar(x - width, rolls, width, label="Roll", color="#4C72B0")
    bars_p = ax.bar(x, pitches, width, label="Pitch", color="#DD8452")
    bars_y = ax.bar(x + width, yaws, width, label="Yaw", color="#55A868")
    ax.set_xlabel("Trip")
    ax.set_ylabel("Angle Delta (deg, LiDAR frame)")
    ax.set_title("RPY Calibration Delta per Trip (Original vs Calibrated)")
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
    ax.legend()
    ax.axhline(y=0, color="gray", linewidth=0.5)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fname = "rpy_delta_comparison.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=120)
    plt.close(fig)
    charts.append(fname)

    fig, ax = plt.subplots(figsize=(max(8, len(ok_results) * 1.5), 5))
    ax.bar(x - width, roll_stds, width, label="Roll std", color="#4C72B0")
    ax.bar(x, pitch_stds, width, label="Pitch std", color="#DD8452")
    ax.bar(x + width, yaw_stds, width, label="Yaw std", color="#55A868")
    ax.axhline(y=0.05, color="red", linewidth=1, linestyle="--", label="HIGH threshold")
    ax.axhline(y=0.15, color="orange", linewidth=1, linestyle="--", label="MEDIUM threshold")
    ax.set_xlabel("Trip")
    ax.set_ylabel("Std (deg)")
    ax.set_title("Per-frame Prediction Consistency (lower = more confident)")
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fname2 = "confidence_std_comparison.png"
    fig.savefig(os.path.join(output_dir, fname2), dpi=120)
    plt.close(fig)
    charts.append(fname2)

    fig, ax = plt.subplots(figsize=(max(8, len(ok_results) * 1.2), 4))
    ax.bar(x, totals, 0.5, color="#C44E52")
    ax.set_xlabel("Trip")
    ax.set_ylabel("Total Rotation Delta (deg)")
    ax.set_title("Geodesic Rotation Adjustment per Trip")
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fname3 = "total_rot_delta.png"
    fig.savefig(os.path.join(output_dir, fname3), dpi=120)
    plt.close(fig)
    charts.append(fname3)

    if len(ok_results) >= 2:
        lines = []
        lines.append("## RPY Statistics Across All Trips (LiDAR frame)")
        lines.append("")
        lines.append("| Stat | Roll (deg) | Pitch (deg) | Yaw (deg) | Total (deg) |")
        lines.append("| --- | --- | --- | --- | --- |")
        for label, arr in [("Mean", [rolls, pitches, yaws, totals]),
                           ("Std", None), ("Min", None), ("Max", None)]:
            if label == "Mean":
                vals = [np.mean(rolls), np.mean(pitches), np.mean(yaws), np.mean(totals)]
            elif label == "Std":
                vals = [np.std(rolls), np.std(pitches), np.std(yaws), np.std(totals)]
            elif label == "Min":
                vals = [np.min(rolls), np.min(pitches), np.min(yaws), np.min(totals)]
            elif label == "Max":
                vals = [np.max(rolls), np.max(pitches), np.max(yaws), np.max(totals)]
            lines.append("| {} | {:.6f} | {:.6f} | {:.6f} | {:.6f} |".format(label, *vals))
        lines.append("")

        stats_path = os.path.join(output_dir, "_rpy_stats.md")
        with io.open(stats_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    return charts


# =============================================================================
# Section: Child JSON worker (multi-GPU subprocess isolation)
# =============================================================================
def run_trip_job_from_json(job_path):
    """
    Execute a single trip calibration described by JSON (used by multi-GPU workers).

    JSON schema (keys):
        trip_name, trip_dir, bag_paths, config_dir, kwargs (dict of remaining options)
    """
    with io.open(job_path, "r", encoding="utf-8") as handle:
        job = json.load(handle)
    trip_name = job["trip_name"]
    trip_dir = job.get("trip_dir")
    bag_paths = job.get("bag_paths")
    config_dir = job.get("config_dir")
    kwargs = dict(job.get("kwargs", {}))
    if isinstance(kwargs.get("img_shape"), list):
        kwargs["img_shape"] = (int(kwargs["img_shape"][0]), int(kwargs["img_shape"][1]))
    return calibrate_trip(
        trip_name=trip_name,
        trip_dir=trip_dir,
        bag_paths=bag_paths,
        config_dir=config_dir,
        **kwargs
    )


def write_trip_job(tmp_root, job_dict):
    os.makedirs(tmp_root, exist_ok=True)
    path = os.path.join(tmp_root, "job_{}.json".format(uuid.uuid4().hex))
    with io.open(path, "w", encoding="utf-8") as handle:
        json.dump(job_dict, handle, indent=2)
    return path


# =============================================================================
# Section: Air suspension filter (空悬档位检测)
# =============================================================================

_AIR_SUSP_FIELD_NUM = 46
_AIR_SUSP_NORMAL_VALUE = 0
_AIR_SUSP_SAMPLE_INTERVAL = 10


def _extract_air_susp_timestamps(bag_dir):
    """Scan bags for /canbus/car_state and return dict of timestamps (sec) -> bool.

    True = normal/pass, False = abnormal air suspension level.
    Empty dict if no car_state data found (all frames pass).

    Uses rosbags (pure Python) to read bags. VehicleState proto has a ``$$$$``
    header wrapper; the actual protobuf body follows after the 8-byte magic + hdr_len.
    Field 46 (air_susp_report) sub-field 1 (air_susp_lvl enum): NORMAL = 0.
    """
    try:
        from rosbags.rosbag1 import Reader
    except ImportError:
        print("[air_susp] rosbags not installed, skipping filter")
        return {}

    car_state_map = {}
    bags_dir = bag_dir if isinstance(bag_dir, str) else str(bag_dir)
    total_msgs = 0
    abnormal_count = 0

    sample_interval = _AIR_SUSP_SAMPLE_INTERVAL
    max_bags_to_scan = 3
    bags_scanned = 0
    for root, _, files in os.walk(bags_dir):
        if "Heavy_Topic_Group" in root or "Medium_Topic_Group" in root:
            continue
        for f in sorted(files):
            if not f.endswith(".bag") or bags_scanned >= max_bags_to_scan:
                continue
            bag_path = os.path.join(root, f)
            try:
                with Reader(bag_path) as reader:
                    cs_conns = [c for c in reader.connections
                                if c.topic == "/canbus/car_state"]
                    if not cs_conns:
                        continue
                    bags_scanned += 1
                    msg_idx = 0
                    for conn, timestamp, rawdata in reader.messages(connections=cs_conns):
                        msg_idx += 1
                        total_msgs += 1
                        if msg_idx % sample_interval != 0:
                            continue
                        ts = timestamp / 1e9
                        body = _strip_vehicle_state_header(rawdata)
                        is_normal = _check_air_susp_from_body(body)
                        car_state_map[ts] = is_normal
                        if not is_normal:
                            abnormal_count += 1
            except Exception:
                continue

    if total_msgs > 0:
        print("[air_susp] Scanned {} car_state msgs, {} abnormal ({:.1f}%)".format(
            total_msgs, abnormal_count, abnormal_count / total_msgs * 100))
    return car_state_map


def _strip_vehicle_state_header(rawdata):
    """Strip std_msgs/String wrapper and $$$$ header from raw bag data."""
    if len(rawdata) < 8:
        return rawdata
    str_len = struct.unpack("<I", rawdata[:4])[0]
    proto_data = rawdata[4:4 + str_len]
    if proto_data[:4] == b"\x24\x24\x24\x24":
        hdr_len = struct.unpack("<I", proto_data[4:8])[0]
        return proto_data[8 + hdr_len:]
    return proto_data


def _check_air_susp_from_body(body):
    """Check air_susp_lvl from VehicleState protobuf body.

    Field 46 = air_susp_report (sub-message).
    Sub-field 1 = air_susp_lvl (enum): NORMAL = 0.
    If field 46 absent -> True (pass). If present and lvl == NORMAL -> True.
    """
    from google.protobuf.internal.decoder import _DecodeVarint32

    pos = 0
    end = len(body)
    while pos < end:
        try:
            tag, new_pos = _DecodeVarint32(body, pos)
        except Exception:
            break
        fn = tag >> 3
        wt = tag & 0x7
        if wt == 2:
            length, new_pos = _DecodeVarint32(body, new_pos)
            if fn == _AIR_SUSP_FIELD_NUM:
                sub = body[new_pos:new_pos + length]
                return _read_susp_lvl(sub) == _AIR_SUSP_NORMAL_VALUE
            pos = new_pos + length
        elif wt == 0:
            _, pos = _DecodeVarint32(body, new_pos)
        elif wt == 1:
            pos = new_pos + 8
        elif wt == 5:
            pos = new_pos + 4
        else:
            break
    return True


def _read_susp_lvl(sub_bytes):
    """Read air_susp_lvl (sub-field 1, varint) from AirSuspReport sub-message."""
    from google.protobuf.internal.decoder import _DecodeVarint32

    pos = 0
    end = len(sub_bytes)
    while pos < end:
        try:
            tag, new_pos = _DecodeVarint32(sub_bytes, pos)
        except Exception:
            break
        fn = tag >> 3
        wt = tag & 0x7
        if wt == 0:
            value, new_pos = _DecodeVarint32(sub_bytes, new_pos)
            if fn == 1:
                return value
            pos = new_pos
        elif wt == 2:
            length, new_pos = _DecodeVarint32(sub_bytes, new_pos)
            pos = new_pos + length
        elif wt == 1:
            pos = new_pos + 8
        elif wt == 5:
            pos = new_pos + 4
        else:
            break
    return _AIR_SUSP_NORMAL_VALUE


def get_air_susp_status_for_frames(car_state_map, times, tolerance_sec=1.0):
    """Map frame timestamps to air suspension status using nearest car_state message.

    Args:
        car_state_map: dict from _extract_air_susp_timestamps()
        times: list of frame timestamps (seconds)
        tolerance_sec: max time gap for matching

    Returns:
        list of bools, True = normal (pass), False = abnormal (skip)
    """
    if not car_state_map or times is None:
        return [True] * (len(times) if times else 0)

    sorted_cs = sorted(car_state_map.keys())
    result = []
    for t in times:
        if t is None:
            result.append(True)
            continue
        idx = _bisect_nearest(sorted_cs, t)
        if idx < 0 or idx >= len(sorted_cs):
            result.append(True)
            continue
        if abs(sorted_cs[idx] - t) <= tolerance_sec:
            result.append(car_state_map[sorted_cs[idx]])
        else:
            result.append(True)
    return result


def _bisect_nearest(sorted_list, target):
    idx = bisect.bisect_left(sorted_list, target)
    if idx == 0:
        return 0
    if idx == len(sorted_list):
        return len(sorted_list) - 1
    if abs(sorted_list[idx] - target) < abs(sorted_list[idx - 1] - target):
        return idx
    return idx - 1


# =============================================================================
# Section: Core per-trip routine
# =============================================================================
def _compute_instant_speed(poses, idx, times, default_fps=10.0):
    """Return instantaneous speed (m/s) at frame ``idx`` using pose displacement."""
    if poses is None or idx <= 0 or idx >= len(poses):
        return None
    cur = poses[idx][:3, 3]
    prev = poses[idx - 1][:3, 3]
    dt = 1.0 / default_fps
    if times is not None and idx < len(times) and (idx - 1) < len(times):
        t_diff = float(times[idx] - times[idx - 1])
        if t_diff > 0.001:
            dt = t_diff
    return float(np.linalg.norm(cur - prev)) / dt


def _compute_acceleration(poses, idx, times, default_fps=10.0):
    """Return acceleration magnitude (m/s^2) at frame ``idx`` via finite difference of velocity."""
    if poses is None or idx <= 1 or idx >= len(poses):
        return None

    def _dt(i):
        if times is not None and i < len(times) and (i - 1) < len(times):
            d = float(times[i] - times[i - 1])
            if d > 0.001:
                return d
        return 1.0 / default_fps

    dt1 = _dt(idx - 1)
    dt2 = _dt(idx)
    v_prev = float(np.linalg.norm(poses[idx - 1][:3, 3] - poses[idx - 2][:3, 3])) / dt1
    v_curr = float(np.linalg.norm(poses[idx][:3, 3] - poses[idx - 1][:3, 3])) / dt2
    dt_mid = 0.5 * (dt1 + dt2)
    if dt_mid < 0.001:
        return None
    return abs(v_curr - v_prev) / dt_mid


def _check_frame_interval(strategy, interval, idx, last_accepted_idx, poses, times):
    """Return True if frame ``idx`` satisfies the interval condition vs ``last_accepted_idx``."""
    if strategy == "none":
        return True
    if last_accepted_idx < 0:
        return True

    if strategy == "time" or strategy == "adaptive":
        if times is not None and idx < len(times) and last_accepted_idx < len(times):
            return float(times[idx] - times[last_accepted_idx]) >= interval
        return True

    if strategy == "distance":
        if poses is not None and idx < len(poses) and last_accepted_idx < len(poses):
            cur = poses[idx][:3, 3]
            last = poses[last_accepted_idx][:3, 3]
            return float(np.linalg.norm(cur - last)) >= interval
        return True

    return True


def _streaming_extract_legacy(trip_name, trip_dir, config_dir, extract_root,
                              extract_cap, camera_name, preparer_num_workers,
                              force_config):
    """
    Streaming extraction for remote mode — mirrors C++ DPBag View pattern.

    Core idea: the StreamingTripDownloader downloads bags in background threads
    while this function runs extraction concurrently.

    Strategy (pipelined, not sequential):
      1) Snapshot bags that have landed so far.
      2) Symlink them into a flat staging directory.
      3) Launch BEVCalibDatasetPreparer (which processes those bags).
         Meanwhile, background downloads continue adding more bags.
      4) After first extraction pass, check if cap was reached.
      5) If not, collect newly landed bags, add to staging, run another pass.
      6) Repeat until cap is met or all downloads finish.
      7) Final sync_and_save, then stop downloader.

    This overlaps download I/O with extraction CPU work, achieving true
    "边下载边标定" concurrency.
    """
    downloader = _ACTIVE_DOWNLOADERS[trip_name]
    staging = os.path.join(extract_root, "bag_staging")
    os.makedirs(staging, exist_ok=True)
    seen_bags = set()

    def _snapshot_new_bags():
        """Find newly landed bags and symlink into staging (skip .__tmp__ partials)."""
        bags_dir = os.path.join(trip_dir, "bags")
        newly = []
        if os.path.isdir(bags_dir):
            for root, _, files in os.walk(bags_dir):
                for f in files:
                    if not f.endswith(".bag") or f.startswith(".__tmp__"):
                        continue
                    full = os.path.join(root, f)
                    if full in seen_bags:
                        continue
                    try:
                        if os.path.getsize(full) < 1000:
                            continue
                    except OSError:
                        continue
                    link_dst = os.path.join(staging, f)
                    if not os.path.exists(link_dst):
                        os.symlink(full, link_dst)
                    seen_bags.add(full)
                    newly.append(f)
        return newly

    total_queued = len(downloader._download_queue)

    # Phase 1: Wait for 60% of bags to land before starting extraction.
    # With download priority (Heavy first), 60% means most Heavy bags are ready.
    target_ready = max(4, int(total_queued * 0.6))
    t_wait_start = time.time()
    while downloader.downloaded_count < target_ready:
        if time.time() - t_wait_start > 180:
            break
        if downloader._stop.is_set():
            break
        _snapshot_new_bags()
        time.sleep(2)

    _snapshot_new_bags()
    n_staged = len([f for f in os.listdir(staging) if f.endswith(".bag")])
    if n_staged == 0:
        print("[stream] WARNING: no bags available for extraction")
        downloader.stop()
        if trip_name in _ACTIVE_DOWNLOADERS:
            del _ACTIVE_DOWNLOADERS[trip_name]
        return

    # Phase 2: Extract from available bags (downloads continue in background)
    _prep_workers = min(preparer_num_workers, max(4, n_staged))
    preparer = BEVCalibDatasetPreparer(
        bag_path=staging, config_dir=config_dir, output_dir=extract_root,
        camera_name=camera_name, target_fps=10.0, max_time_diff=0.055,
        batch_size=500, num_workers=_prep_workers, max_frames=extract_cap,
        save_debug_samples=0, max_pose_gap=0.5, force_config=force_config,
        sequence_id="00",
    )
    print("[stream] Phase 1: extracting from {} bags (dl {}/{}, cap={})".format(
        n_staged, downloader.downloaded_count, total_queued, extract_cap))
    preparer.extract_data_from_bag()
    preparer.sync_and_save(sequence_id="00")

    seq_img = os.path.join(extract_root, "sequences", "00", "image_2")
    n_frames = len(os.listdir(seq_img)) if os.path.isdir(seq_img) else 0
    print("[stream] Phase 1 done: {} frames from {} bags".format(n_frames, n_staged))

    # Phase 2: If not enough frames AND more bags arrived, extract ONLY new bags
    if n_frames < extract_cap:
        dl_done = downloader._stop.is_set() or \
            downloader.downloaded_count >= total_queued
        if not dl_done:
            print("[stream] Waiting for remaining downloads...")
            downloader.wait_for_min_bags(total_queued, timeout=180)

        new_names_phase2 = _snapshot_new_bags()
        if new_names_phase2:
            staging2 = os.path.join(extract_root, "bag_staging_p2")
            os.makedirs(staging2, exist_ok=True)
            for f in new_names_phase2:
                src = os.path.join(staging, f)
                if os.path.exists(src):
                    dst = os.path.join(staging2, f)
                    if not os.path.exists(dst):
                        os.symlink(os.path.realpath(src), dst)
            n_new = len(new_names_phase2)
            _prep_workers = min(preparer_num_workers, max(4, n_new))
            remaining_cap = max(1, extract_cap - n_frames)
            preparer2 = BEVCalibDatasetPreparer(
                bag_path=staging2, config_dir=config_dir, output_dir=extract_root,
                camera_name=camera_name, target_fps=10.0, max_time_diff=0.055,
                batch_size=500, num_workers=_prep_workers, max_frames=remaining_cap,
                save_debug_samples=0, max_pose_gap=0.5, force_config=force_config,
                sequence_id="00",
            )
            print("[stream] Phase 2: extracting from {} NEW bags only (cap={})".format(
                n_new, remaining_cap))
            preparer2.extract_data_from_bag()
            preparer2.sync_and_save(sequence_id="00")
            n_frames = len(os.listdir(seq_img)) if os.path.isdir(seq_img) else 0
            print("[stream] Phase 2 done: {} frames total".format(n_frames))

    downloader.stop()
    if trip_name in _ACTIVE_DOWNLOADERS:
        del _ACTIVE_DOWNLOADERS[trip_name]


def _streaming_calibrate_loop(
    trip_name, trip_dir, config_dir, extract_root,
    max_frames, extract_buffer_multiplier, camera_name,
    preparer_num_workers, force_config,
    wrapper, T_init, K_full, img_shape, device, batch_size,
    xbound, ybound, zbound,
    min_speed_kmh, max_speed_kmh, min_accel, max_accel, min_brightness,
    cv2, torch,
    filter_air_suspension=False,
    infer_cap_multiplier=1.25,
):
    """Async-pipelined streaming calibration: download | extract | infer run concurrently.

    Three-stage pipeline:
      Stage 1 (background): StreamingTripDownloader continuously downloads bags
      Stage 2 (thread):     BEVCalibDatasetPreparer extracts frames from landed bags
      Stage 3 (main):       Quality filter + I/O prefetch + GPU inference

    Extraction runs in a background thread so it overlaps with inference.
    I/O (image + pointcloud loading) is prefetched via ThreadPoolExecutor while
    the GPU processes the current batch.

    Returns:
        (inferred_predictions, inferred_frame_meta, qf_stats, t_extract_sec, t_infer_sec)
    """
    _ = K_full
    downloader = _ACTIVE_DOWNLOADERS[trip_name]
    staging = os.path.join(extract_root, "bag_staging")
    os.makedirs(staging, exist_ok=True)
    seen_bags = set()
    processed_stems = set()
    scanned_air_susp_bags = set()
    air_susp_map = {}

    inferred_predictions = []
    inferred_frame_meta = []
    qf_stats = {"skip_static": 0, "skip_fast": 0, "skip_dark": 0,
                "skip_points": 0, "skip_accel": 0, "skip_air_susp": 0,
                "total_scanned": 0, "inferred": 0}

    min_speed_ms = min_speed_kmh / 3.6
    max_speed_ms = max_speed_kmh / 3.6
    Hm, Wm = int(img_shape[0]), int(img_shape[1])
    scale_x = None
    scale_y = None
    infer_device = torch.device(device)

    t_extract_total = 0.0
    t_infer_total = 0.0
    round_num = 0
    max_rounds = 50
    max_infer_cap = max(max_frames + 1, int(max_frames * infer_cap_multiplier))

    _extract_lock = threading.Lock()
    _extract_done_event = threading.Event()
    _extract_error = [None]
    _bg_extract_running = [False]

    def _snapshot_new_bags():
        if isinstance(downloader, LocalBagGate):
            return downloader.snapshot_new_bags(staging, seen_bags)
        bags_dir = os.path.join(trip_dir, "bags")
        newly = []
        if os.path.isdir(bags_dir):
            for root, _, files in os.walk(bags_dir):
                for f in files:
                    if not f.endswith(".bag") or f.startswith(".__tmp__"):
                        continue
                    full = os.path.join(root, f)
                    if full in seen_bags:
                        continue
                    try:
                        if os.path.getsize(full) < 1000:
                            continue
                    except OSError:
                        continue
                    link_dst = os.path.join(staging, f)
                    if not os.path.exists(link_dst):
                        os.symlink(full, link_dst)
                    seen_bags.add(full)
                    newly.append(f)
        return newly

    def _has_heavy_and_medium():
        heavy = medium = False
        for f in os.listdir(staging) if os.path.isdir(staging) else []:
            if "Heavy_Topic_Group" in f:
                heavy = True
            if "Medium_Topic_Group" in f:
                medium = True
            if heavy and medium:
                return True
        return False

    def _bg_extract(n_staged, remaining_cap):
        """Background extraction thread: extract frames from staged bags."""
        try:
            _prep_workers = min(preparer_num_workers, max(4, n_staged))
            preparer = BEVCalibDatasetPreparer(
                bag_path=staging, config_dir=config_dir, output_dir=extract_root,
                camera_name=camera_name, target_fps=10.0, max_time_diff=0.055,
                batch_size=500, num_workers=_prep_workers,
                max_frames=remaining_cap + len(processed_stems),
                save_debug_samples=0, max_pose_gap=0.5, force_config=force_config,
                sequence_id="00",
            )
            preparer.extract_data_from_bag()
            preparer.sync_and_save(sequence_id="00")
        except (ValueError, RuntimeError) as exc:
            _extract_error[0] = exc
        finally:
            _bg_extract_running[0] = False
            _extract_done_event.set()

    def _load_frame_data(stem, seq_dir, K_full_local, sx, sy):
        """Load image + pointcloud for a single frame (runs in I/O thread pool)."""
        img_path = os.path.join(seq_dir, "image_2", stem + ".png")
        if not os.path.isfile(img_path):
            img_path = os.path.join(seq_dir, "image_2", stem + ".jpg")

        gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if gray is not None and float(np.mean(gray)) < min_brightness:
            return "skip_dark", None

        im = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if im is None:
            return "skip_read", None

        ih, iw = im.shape[:2]
        local_sx = sx if sx is not None else float(Wm) / float(iw)
        local_sy = sy if sy is not None else float(Hm) / float(ih)

        pc_path = os.path.join(seq_dir, "velodyne", stem + ".bin")
        pc_raw = load_raw_pointcloud(pc_path)
        pc_f = filter_pointcloud_for_bev(pc_raw, xbound, ybound, zbound)
        if pc_f.shape[0] < 10:
            return "skip_points", None

        Ks = K_full_local.copy()
        Ks[0, 0] *= local_sx
        Ks[0, 2] *= local_sx
        Ks[1, 1] *= local_sy
        Ks[1, 2] *= local_sy
        im_r = cv2.resize(im, (Wm, Hm), interpolation=cv2.INTER_AREA)
        return "ok", (im_r, pc_f[:, :3].astype(np.float32), Ks, img_path, pc_path,
                       local_sx, local_sy)

    if isinstance(downloader, LocalBagGate):
        _snapshot_new_bags()
        n_initial = len([f for f in os.listdir(staging) if f.endswith(".bag")]) if os.path.isdir(staging) else 0
        print("[stream] Local incremental gate: {} bags in first batch ({} slots total)".format(
            n_initial, len(downloader._slots)))
    else:
        print("[stream] Waiting for initial Heavy+Medium bags to arrive...")
        t_wait = time.time()
        while not _has_heavy_and_medium():
            elapsed = time.time() - t_wait
            if elapsed > 90:
                print("[stream] WARNING: timeout waiting for Heavy+Medium bags ({:.0f}s)".format(elapsed))
                break
            if downloader._stop.is_set():
                break
            _snapshot_new_bags()
            time.sleep(0.5)
        _snapshot_new_bags()
        n_initial = len([f for f in os.listdir(staging) if f.endswith(".bag")]) if os.path.isdir(staging) else 0
        print("[stream] Initial bags ready: {} in staging".format(n_initial))

    if downloader.has_unqueued_slots:
        downloader.request_more_bags(3)

    while round_num < max_rounds and len(inferred_predictions) < max_infer_cap:
        round_num += 1

        new_bags = _snapshot_new_bags()

        if not new_bags and round_num > 1:
            if downloader.has_unqueued_slots:
                downloader.request_more_bags(3)
                time.sleep(1)
                continue
            elif downloader.all_downloaded:
                print("[stream] All bags exhausted after {} rounds".format(round_num))
                break
            else:
                time.sleep(1)
                continue

        n_staged = len([f for f in os.listdir(staging) if f.endswith(".bag")])
        if n_staged == 0:
            time.sleep(1)
            continue

        t_ext_start = time.time()
        buf = min(float(extract_buffer_multiplier), 1.3)
        remaining_cap = max(30, int(max_frames * buf) - len(processed_stems))

        _extract_done_event.clear()
        _extract_error[0] = None
        _bg_extract_running[0] = True
        extract_thread = threading.Thread(
            target=_bg_extract, args=(n_staged, remaining_cap), daemon=False)
        extract_thread.start()

        if downloader.has_unqueued_slots:
            downloader.request_more_bags(3)

        extract_timeout = max(300, n_staged * 5)
        _extract_done_event.wait(timeout=extract_timeout)
        extract_thread.join(timeout=60)
        t_extract_total += time.time() - t_ext_start

        if _extract_error[0] is not None:
            print("[stream] Round {} extraction failed ({}), waiting for more bags...".format(
                round_num, _extract_error[0]))
            if downloader.has_unqueued_slots:
                downloader.request_more_bags(3)
            time.sleep(1)
            continue

        seq_dir = os.path.join(extract_root, "sequences", "00")
        img_dir = os.path.join(seq_dir, "image_2")
        vel_dir = os.path.join(seq_dir, "velodyne")
        if not os.path.isdir(img_dir) or not os.path.isdir(vel_dir):
            print("[stream] Round {}: extraction produced no synced frames, requesting more bags".format(round_num))
            if downloader.has_unqueued_slots:
                downloader.request_more_bags(3)
            time.sleep(1)
            continue

        stems = list_sequence_frames(seq_dir)
        new_stems = [s for s in stems if s not in processed_stems]

        if not new_stems:
            if downloader.has_unqueued_slots:
                downloader.request_more_bags(3)
                time.sleep(1)
                continue
            elif downloader.all_downloaded:
                break
            else:
                time.sleep(1)
                continue

        poses_path = os.path.join(extract_root, "poses", "00.txt")
        times_path = os.path.join(seq_dir, "times.txt")
        poses = read_kitti_poses(poses_path) if os.path.isfile(poses_path) else None
        times = read_times_file(times_path) if os.path.isfile(times_path) else None
        calib_info = parse_kitti_calib_txt(os.path.join(seq_dir, "calib.txt"))
        K_full_local = calib_info["K"].copy()

        all_stems_sorted = sorted(stems)
        stem_to_idx = {s: i for i, s in enumerate(all_stems_sorted)}

        t_inf_start = time.time()

        if filter_air_suspension:
            for root_d, _, fnames in os.walk(staging):
                for bf in fnames:
                    if not bf.endswith(".bag") or bf in scanned_air_susp_bags:
                        continue
                    scanned_air_susp_bags.add(bf)
                    bp = os.path.join(root_d, bf)
                    partial = _extract_air_susp_timestamps(bp)
                    air_susp_map.update(partial)

        air_susp_frames = get_air_susp_status_for_frames(
            air_susp_map, times, tolerance_sec=0.5) if air_susp_map else None

        qf_passed_stems = []
        for stem in sorted(new_stems):
            processed_stems.add(stem)
            if len(inferred_predictions) + len(qf_passed_stems) >= max_infer_cap:
                break

            frame_idx = stem_to_idx.get(stem, 0)
            qf_stats["total_scanned"] += 1

            if air_susp_frames and frame_idx < len(air_susp_frames) and not air_susp_frames[frame_idx]:
                qf_stats["skip_air_susp"] += 1
                continue

            speed = _compute_instant_speed(poses, frame_idx, times)
            if speed is not None:
                if speed < min_speed_ms:
                    qf_stats["skip_static"] += 1
                    continue
                if speed > max_speed_ms:
                    qf_stats["skip_fast"] += 1
                    continue

            accel = _compute_acceleration(poses, frame_idx, times)
            if accel is not None and (accel < min_accel or accel > max_accel):
                qf_stats["skip_accel"] += 1
                continue

            ts_val = float(times[frame_idx]) if times is not None and frame_idx < len(times) else None
            ts_str = ""
            if ts_val is not None:
                if ts_val > 1e9:
                    _utc8 = datetime.timezone(datetime.timedelta(hours=8))
                    ts_str = datetime.datetime.fromtimestamp(ts_val, tz=_utc8).strftime("%Y-%m-%d %H:%M:%S")
                else:
                    ts_str = "{:.2f}s".format(ts_val)

            qf_passed_stems.append((stem, frame_idx, speed, accel, ts_str, ts_val))

        io_pool = ThreadPoolExecutor(max_workers=4)
        mini_batch_imgs = []
        mini_batch_pcs = []
        mini_batch_Ks = []
        mini_batch_meta = []

        def _flush_mini():
            if not mini_batch_imgs:
                return
            with torch.no_grad():
                batch = collate_and_pad_batch(
                    list(mini_batch_imgs), list(mini_batch_pcs),
                    T_init, list(mini_batch_Ks), infer_device, torch,
                )
                if batch is not None:
                    imgs_t, pcs_t, init_T_t, post_T_t, K_t = batch
                    if device.startswith("cuda") and not getattr(wrapper, "disable_amp", False):
                        with torch.cuda.amp.autocast():
                            preds_t = wrapper(imgs_t, pcs_t, init_T_t, post_T_t, K_t)
                    else:
                        preds_t = wrapper(imgs_t, pcs_t, init_T_t, post_T_t, K_t)
                    preds_np = preds_t.detach().cpu().numpy()
                    for bi in range(preds_np.shape[0]):
                        inferred_predictions.append(preds_np[bi])
                        inferred_frame_meta.append(mini_batch_meta[bi])
                        qf_stats["inferred"] += 1
            mini_batch_imgs.clear()
            mini_batch_pcs.clear()
            mini_batch_Ks.clear()
            mini_batch_meta.clear()

        prefetch_window = min(batch_size * 3, len(qf_passed_stems))
        futures = []
        for i, (stem, _, _, _, _, _) in enumerate(qf_passed_stems[:prefetch_window]):
            futures.append(io_pool.submit(
                _load_frame_data, stem, seq_dir, K_full_local, scale_x, scale_y))

        fi = 0
        for si, (stem, frame_idx, speed, accel, ts_str, ts_val) in enumerate(qf_passed_stems):
            if len(inferred_predictions) >= max_infer_cap:
                break

            if fi < len(futures):
                status, data = futures[fi].result()
                fi += 1
            else:
                status, data = _load_frame_data(
                    stem, seq_dir, K_full_local, scale_x, scale_y)

            next_prefetch = si + prefetch_window
            if next_prefetch < len(qf_passed_stems):
                ns, _, _, _, _, _ = qf_passed_stems[next_prefetch]
                futures.append(io_pool.submit(
                    _load_frame_data, ns, seq_dir, K_full_local, scale_x, scale_y))

            if status == "skip_dark":
                qf_stats["skip_dark"] += 1
                continue
            elif status == "skip_points":
                qf_stats["skip_points"] += 1
                continue
            elif status != "ok" or data is None:
                continue

            im_r, pc_f, Ks, img_path, pc_path, local_sx, local_sy = data
            if scale_x is None:
                scale_x = local_sx
                scale_y = local_sy

            mini_batch_imgs.append(im_r)
            mini_batch_pcs.append(pc_f)
            mini_batch_Ks.append(Ks)
            mini_batch_meta.append({
                "stem": stem, "frame_idx": frame_idx,
                "img_path": img_path, "pc_path": pc_path, "K": Ks,
                "timestamp": ts_str, "timestamp_sec": ts_val,
                "speed": speed, "accel": accel,
            })

            if len(mini_batch_imgs) >= batch_size:
                _flush_mini()

        if mini_batch_imgs:
            _flush_mini()
        io_pool.shutdown(wait=False)

        t_infer_total += time.time() - t_inf_start

        n_acc = len(inferred_predictions)
        print("[stream] Round {}: +{} new frames, total inferred={}/{}, scanned={}".format(
            round_num, len(new_stems), n_acc, max_frames,
            qf_stats["total_scanned"]))

        if n_acc >= max_frames:
            print("[stream] Target reached: {} frames in {} rounds".format(n_acc, round_num))
            break

        if downloader.has_unqueued_slots:
            downloader.request_more_bags(3)
        elif downloader.all_downloaded:
            print("[stream] All bags processed, got {} frames (target {})".format(n_acc, max_frames))
            break

    downloader.stop()
    if trip_name in _ACTIVE_DOWNLOADERS:
        del _ACTIVE_DOWNLOADERS[trip_name]

    print("[stream] Streaming complete: {} rounds, {} frames, extract={:.1f}s, infer={:.1f}s".format(
        round_num, len(inferred_predictions), t_extract_total, t_infer_total))

    return inferred_predictions, inferred_frame_meta, qf_stats, t_extract_total, t_infer_total


def calibrate_trip(
    trip_name,
    trip_dir=None,
    bag_paths=None,
    config_dir=None,
    output_dir=None,
    ckpt_path=None,
    max_frames=400,
    min_agg_frames=50,
    extract_buffer_multiplier=1.5,
    camera_name="traffic_2",
    sample_strategy="time",
    sample_interval=1.0,
    device="cuda:0",
    batch_size=24,
    img_shape=(360, 640),
    projection_ratio=0.1,
    keep_temp=False,
    force_config=False,
    preparer_num_workers=16,
    trip_timeout_sec=None,
    min_speed_kmh=5.0,
    max_speed_kmh=120.0,
    min_accel=-1.0,
    max_accel=2.0,
    min_brightness=25,
    filter_air_suspension=False,
    input_format=None,
    initial_bag_groups=3,
    gt_lidars_cfg=None,
    inject_lidar_rpy_deg=None,
    incremental_extract=True,
    infer_cap_multiplier=1.25,
    multi_init_sweep_deg=0.0,
    multi_init_grid=False,
):
    """
    Streaming calibration pipeline for one trip.

    Architecture: extract raw frames -> iterate one-by-one -> per-frame quality gate
    (static/dark/speed) + interval check -> immediate single-frame inference ->
    cache prediction for aggregation -> stop at max_frames.

    Args:
        projection_ratio (float): Fraction of accepted frames to generate projection images
            (0.1 = 10%, so 400 frames -> 40 projections). 0 disables.
        min_speed_kmh (float): Skip frames with speed < this (km/h). Default 10 km/h.
        max_speed_kmh (float): Skip frames with speed > this (km/h). Default 80 km/h.
        min_accel (float): Skip frames with acceleration < this (m/s^2). Default -0.25.
        max_accel (float): Skip frames with acceleration > this (m/s^2). Default 1.0.
        min_brightness (int): Skip frames with mean gray < this. Default 25.
    """
    _ = trip_timeout_sec

    if ConfigParser is None or BEVCalibDatasetPreparer is None:
        raise RuntimeError("prepare_custom_dataset imports failed — check tools/preparation on PYTHONPATH.")
    if load_bevcalib_inference is None or TemporalCalibrationAggregator is None:
        raise RuntimeError("utils.bevcalib_inference import failed.")

    torch = _require_torch()
    cv2 = _require_cv2()

    trip_out = os.path.join(output_dir, trip_name)
    os.makedirs(trip_out, exist_ok=True)
    proj_dir = os.path.join(trip_out, "projections")
    os.makedirs(proj_dir, exist_ok=True)
    log_path = os.path.join(trip_out, "calibration.log")

    result = {
        "trip": trip_name, "status": "unknown", "frames": 0,
        "rot_delta": float("nan"), "trans_delta": float("nan"),
        "roll_std": float("nan"), "pitch_std": float("nan"),
        "yaw_std": float("nan"), "total_std": float("nan"), "error": "",
    }

    extract_root = tempfile.mkdtemp(prefix="bevcalib_kitti_{}_".format(trip_name))
    t_trip_start = time.time()

    try:
        with tee_stdout_stderr(log_path):
            print("\n{}".format("=" * 80))
            print("Trip: {}  [streaming mode]".format(trip_name))
            print("{}".format("=" * 80))

            if trip_dir and not config_dir:
                config_dir = find_trip_config_dir(trip_dir)
            if config_dir is None:
                raise ValueError("config_dir could not be resolved.")

            cameras_cfg = os.path.join(config_dir, "cameras.cfg")
            lidars_cfg = os.path.join(config_dir, "lidars.cfg")
            if not os.path.isfile(cameras_cfg) or not os.path.isfile(lidars_cfg):
                raise FileNotFoundError("cameras.cfg or lidars.cfg missing in {}".format(config_dir))

            cameras = ConfigParser.parse_cameras_cfg(cameras_cfg)
            lidars = ConfigParser.parse_lidars_cfg(lidars_cfg)
            if camera_name not in cameras:
                raise KeyError("Camera '{}' not in cameras.cfg (have: {})".format(
                    camera_name, list(cameras.keys())))

            cam_cfg = cameras[camera_name]

            gt_lidars_path = _resolve_gt_lidars_cfg_path(gt_lidars_cfg, config_dir)
            T_gt_ref = None
            if gt_lidars_path:
                T_gt_ref = _load_gt_T_from_lidars_cfg(gt_lidars_path, cam_cfg)
                print("[info] True GT from {} (geodesic ref for MEDW / shortcut)".format(
                    gt_lidars_path))

            if inject_lidar_rpy_deg:
                rpy_off = (_parse_rpy_triplet(inject_lidar_rpy_deg)
                           if isinstance(inject_lidar_rpy_deg, str)
                           else tuple(inject_lidar_rpy_deg))
                lidars = _apply_lidar_rpy_offset(lidars, rpy_off)
                print("[info] Applied inject_lidar_rpy_deg=({:+.3f}, {:+.3f}, {:+.3f}) on sensor_to_lidar".format(
                    *rpy_off))

            T_init = compute_T_lidar_to_cam(cam_cfg, lidars)
            if T_gt_ref is not None:
                inj = T_to_calibration_metrics(T_gt_ref, T_init)
                print("[info] Injected bias vs true GT: {:.4f} deg "
                      "(R{:+.3f} P{:+.3f} Y{:+.3f})".format(
                          inj["rot_geodesic_deg"],
                          inj["roll_delta_deg"], inj["pitch_delta_deg"], inj["yaw_delta_deg"]))

            extract_cap = int(max(1, max_frames * min(extract_buffer_multiplier, 1.3)))

            if trip_name not in _ACTIVE_DOWNLOADERS and input_format == "remote" and trip_dir:
                bag_staging = os.path.join(trip_dir, "bags", "important")
                if os.path.isdir(bag_staging):
                    trips_base = os.path.dirname(trip_dir)
                    dl = StreamingTripDownloader(
                        trip_name, trips_base,
                        initial_bag_groups=initial_bag_groups)
                    dl.prepare()
                    if dl._download_queue:
                        dl.start_downloads()
                        dl.wait_for_min_bags(min_count=4, timeout=300)
                    _ACTIVE_DOWNLOADERS[trip_name] = dl
                    print("[info] Subprocess created streaming downloader for {}".format(trip_name))

            if (trip_name not in _ACTIVE_DOWNLOADERS and trip_dir
                    and incremental_extract and input_format in ("trips", "remote")):
                register_local_incremental_gate(
                    trip_name, trip_dir, initial_bag_groups=initial_bag_groups)

            _is_streaming = trip_name in _ACTIVE_DOWNLOADERS

            xbound, ybound, zbound = _lazy_bev_bounds()
            Hm, Wm = int(img_shape[0]), int(img_shape[1])

            wrapper, epoch = load_bevcalib_inference(ckpt_path, device=device, img_shape=(Hm, Wm))
            print("[info] Loaded checkpoint epoch={} device={}".format(epoch, device))

            min_speed_ms = min_speed_kmh / 3.6
            max_speed_ms = max_speed_kmh / 3.6
            qf_stats = {"skip_static": 0, "skip_fast": 0, "skip_dark": 0,
                         "skip_points": 0, "skip_accel": 0, "skip_air_susp": 0,
                         "total_scanned": 0, "inferred": 0}

            inferred_predictions = []
            inferred_frame_meta = []
            max_infer_cap = max(max_frames + 1, int(max_frames * infer_cap_multiplier))
            infer_device = torch.device(device)
            K_full = None

            print("[info] Inference: max_frames={}, max_infer_cap={}, "
                  "speed=[{:.1f}, {:.1f}] km/h, accel=[{:.2f}, {:.2f}] m/s^2, "
                  "batch_size={}, min_brightness={}".format(
                      max_frames, max_infer_cap,
                      min_speed_kmh, max_speed_kmh, min_accel, max_accel,
                      batch_size, min_brightness))

            if _is_streaming:
                inferred_predictions, inferred_frame_meta, qf_stats, t_extract_sec, t_infer_sec = \
                    _streaming_calibrate_loop(
                        trip_name, trip_dir, config_dir, extract_root,
                        max_frames, extract_buffer_multiplier, camera_name,
                        preparer_num_workers, force_config,
                        wrapper, T_init, K_full, img_shape, device, batch_size,
                        xbound, ybound, zbound,
                        min_speed_kmh, max_speed_kmh, min_accel, max_accel, min_brightness,
                        cv2, torch,
                        filter_air_suspension=filter_air_suspension,
                        infer_cap_multiplier=infer_cap_multiplier,
                    )
                n_inferred = len(inferred_predictions)
                print("[info] Data extraction took {:.1f}s".format(t_extract_sec))
                print("[info] Streaming inference done in {:.1f}s: inferred={} scanned={} "
                      "(skip: static={} fast={} accel={} dark={} points={} air_susp={})".format(
                          t_infer_sec, n_inferred, qf_stats["total_scanned"],
                          qf_stats["skip_static"], qf_stats["skip_fast"], qf_stats["skip_accel"],
                          qf_stats["skip_dark"], qf_stats["skip_points"],
                          qf_stats.get("skip_air_susp", 0)))
            else:
                t_extract_start = time.time()
                if bag_paths is None:
                    bag_paths = find_trip_bag_paths(trip_dir)
                bag_paths = subsample_bags_for_calibration(
                    bag_paths, target_frames=max_frames * extract_buffer_multiplier)
                staging = os.path.join(extract_root, "bag_staging")
                bag_root = stage_bags_symlink(bag_paths, staging)
                _num_staged_bags = len([f for f in os.listdir(bag_root) if f.endswith(".bag")])
                _prep_workers = min(preparer_num_workers, max(4, _num_staged_bags // 4))
                preparer = BEVCalibDatasetPreparer(
                    bag_path=bag_root, config_dir=config_dir, output_dir=extract_root,
                    camera_name=camera_name, target_fps=10.0, max_time_diff=0.055,
                    batch_size=500, num_workers=_prep_workers, max_frames=extract_cap,
                    save_debug_samples=0, max_pose_gap=0.5, force_config=force_config,
                    sequence_id="00",
                )
                print("[info] Staged {} bags (subsampled), preparer workers={}, extract_cap={}".format(
                    _num_staged_bags, _prep_workers, extract_cap))
                preparer.extract_data_from_bag()
                preparer.sync_and_save(sequence_id="00")

                t_extract_sec = time.time() - t_extract_start
                print("[info] Data extraction took {:.1f}s".format(t_extract_sec))

                seq_dir = os.path.join(extract_root, "sequences", "00")
                calib_info = parse_kitti_calib_txt(os.path.join(seq_dir, "calib.txt"))
                K_full = calib_info["K"].copy()

                stems = list_sequence_frames(seq_dir)
                print("[info] Extracted {} synced frames from bags".format(len(stems)))

                poses_path = os.path.join(extract_root, "poses", "00.txt")
                times_path = os.path.join(seq_dir, "times.txt")
                poses = read_kitti_poses(poses_path) if os.path.isfile(poses_path) else None
                times = read_times_file(times_path) if os.path.isfile(times_path) else None

                air_susp_map = {}
                air_susp_active = False
                if filter_air_suspension:
                    print("[info] Scanning bags for air suspension status...")
                    air_susp_map = _extract_air_susp_timestamps(bag_root)
                    if air_susp_map:
                        air_susp_frames = get_air_susp_status_for_frames(
                            air_susp_map, times, tolerance_sec=1.0)
                        abnormal_ratio = sum(1 for x in air_susp_frames if not x) / max(len(air_susp_frames), 1)
                        if abnormal_ratio > 0.5:
                            print("[air_susp] WARNING: {:.0f}% frames non-NORMAL. "
                                  "Disabling hard filter (tag-only mode).".format(abnormal_ratio * 100))
                            air_susp_active = False
                        else:
                            air_susp_active = True
                    else:
                        air_susp_frames = [True] * len(stems)
                else:
                    air_susp_frames = [True] * len(stems) if stems else []

                scale_x = None
                scale_y = None

                t_start_infer = time.time()

                _io_pool = ThreadPoolExecutor(max_workers=4)

                def _load_frame_io(stem_, frame_idx_, img_path_, pc_path_):
                    """I/O-bound: load image + pointcloud + quality check (runs in thread)."""
                    gray = cv2.imread(img_path_, cv2.IMREAD_GRAYSCALE)
                    if gray is not None and float(np.mean(gray)) < min_brightness:
                        return ("skip_dark", None)
                    im = cv2.imread(img_path_, cv2.IMREAD_COLOR)
                    if im is None:
                        return ("skip_read", None)
                    pc_raw = load_raw_pointcloud(pc_path_)
                    pc_f = filter_pointcloud_for_bev(pc_raw, xbound, ybound, zbound)
                    if pc_f.shape[0] < 10:
                        return ("skip_points", None)
                    return ("ok", (im, pc_f, stem_, frame_idx_, img_path_, pc_path_))

                prefetch_queue = []
                _prefetch_idx = [0]

                def _submit_prefetch(count):
                    """Submit up to `count` I/O tasks to the thread pool."""
                    submitted = 0
                    while submitted < count and _prefetch_idx[0] < len(stems):
                        fi = _prefetch_idx[0]
                        _prefetch_idx[0] += 1
                        s = stems[fi]
                        if len(inferred_predictions) + len(prefetch_queue) >= max_infer_cap:
                            break
                        qf_stats["total_scanned"] += 1
                        if air_susp_active and fi < len(air_susp_frames) and not air_susp_frames[fi]:
                            qf_stats["skip_air_susp"] += 1
                            continue
                        speed = _compute_instant_speed(poses, fi, times)
                        if speed is not None:
                            if speed < min_speed_ms:
                                qf_stats["skip_static"] += 1
                                continue
                            if speed > max_speed_ms:
                                qf_stats["skip_fast"] += 1
                                continue
                        accel = _compute_acceleration(poses, fi, times)
                        if accel is not None and (accel < min_accel or accel > max_accel):
                            qf_stats["skip_accel"] += 1
                            continue
                        ip = os.path.join(seq_dir, "image_2", s + ".png")
                        if not os.path.isfile(ip):
                            ip = os.path.join(seq_dir, "image_2", s + ".jpg")
                        pp = os.path.join(seq_dir, "velodyne", s + ".bin")
                        ts_val = float(times[fi]) if times is not None and fi < len(times) else None
                        fut = _io_pool.submit(_load_frame_io, s, fi, ip, pp)
                        prefetch_queue.append((fut, fi, s, speed, accel, ts_val))
                        submitted += 1

                _submit_prefetch(batch_size * 3)

                mini_batch_imgs = []
                mini_batch_pcs = []
                mini_batch_Ks = []
                mini_batch_meta = []

                def _flush_mini_batch():
                    if not mini_batch_imgs:
                        return
                    with torch.no_grad():
                        batch = collate_and_pad_batch(
                            list(mini_batch_imgs), list(mini_batch_pcs),
                            T_init, list(mini_batch_Ks), infer_device, torch,
                        )
                        if batch is not None:
                            imgs_t, pcs_t, init_T_t, post_T_t, K_t = batch
                            if device.startswith("cuda") and not getattr(wrapper, "disable_amp", False):
                                with torch.cuda.amp.autocast():
                                    preds_t = wrapper(imgs_t, pcs_t, init_T_t, post_T_t, K_t)
                            else:
                                preds_t = wrapper(imgs_t, pcs_t, init_T_t, post_T_t, K_t)
                            preds_np = preds_t.detach().cpu().numpy()
                            for bi in range(preds_np.shape[0]):
                                inferred_predictions.append(preds_np[bi])
                                inferred_frame_meta.append(mini_batch_meta[bi])
                                qf_stats["inferred"] += 1
                                if qf_stats["inferred"] % 100 == 0:
                                    print("[info] Inferred {}/{} frames (scanned {})".format(
                                        qf_stats["inferred"], max_infer_cap,
                                        qf_stats["total_scanned"]))
                    mini_batch_imgs.clear()
                    mini_batch_pcs.clear()
                    mini_batch_Ks.clear()
                    mini_batch_meta.clear()

                while prefetch_queue and len(inferred_predictions) < max_infer_cap:
                    fut, fi, stem, speed, accel, ts_val = prefetch_queue.pop(0)
                    status, payload = fut.result()
                    if status == "skip_dark":
                        qf_stats["skip_dark"] += 1
                        _submit_prefetch(1)
                        continue
                    if status == "skip_points":
                        qf_stats["skip_points"] += 1
                        _submit_prefetch(1)
                        continue
                    if status != "ok" or payload is None:
                        _submit_prefetch(1)
                        continue
                    im, pc_f, stem, frame_idx, img_path, pc_path = payload
                    ih, iw = im.shape[:2]
                    if scale_x is None:
                        scale_x = float(Wm) / float(iw)
                        scale_y = float(Hm) / float(ih)
                    Ks = K_full.copy()
                    Ks[0, 0] *= scale_x
                    Ks[0, 2] *= scale_x
                    Ks[1, 1] *= scale_y
                    Ks[1, 2] *= scale_y
                    im_r = cv2.resize(im, (Wm, Hm), interpolation=cv2.INTER_AREA)

                    if ts_val is not None:
                        if ts_val > 1e9:
                            _utc8 = datetime.timezone(datetime.timedelta(hours=8))
                            ts_str = datetime.datetime.fromtimestamp(ts_val, tz=_utc8).strftime("%Y-%m-%d %H:%M:%S")
                        else:
                            ts_str = "{:.2f}s".format(ts_val)
                    else:
                        ts_str = ""

                    mini_batch_imgs.append(im_r)
                    mini_batch_pcs.append(pc_f[:, :3].astype(np.float32))
                    mini_batch_Ks.append(Ks)
                    mini_batch_meta.append({
                        "stem": stem, "frame_idx": frame_idx,
                        "img_path": img_path, "pc_path": pc_path, "K": Ks,
                        "timestamp": ts_str, "timestamp_sec": ts_val,
                        "speed": speed, "accel": accel,
                    })

                    if len(mini_batch_imgs) >= batch_size:
                        _flush_mini_batch()
                        _submit_prefetch(batch_size)

                if mini_batch_imgs:
                    _flush_mini_batch()
                _io_pool.shutdown(wait=False)

                t_infer_sec = time.time() - t_start_infer
                n_inferred = len(inferred_predictions)
                print("[info] Inference done in {:.1f}s: inferred={} scanned={} "
                      "(skip: static={} fast={} accel={} dark={} points={} air_susp={})".format(
                          t_infer_sec, n_inferred, qf_stats["total_scanned"],
                          qf_stats["skip_static"], qf_stats["skip_fast"], qf_stats["skip_accel"],
                          qf_stats["skip_dark"], qf_stats["skip_points"],
                          qf_stats.get("skip_air_susp", 0)))

            if n_inferred == 0:
                raise RuntimeError("No frames accepted after quality filtering.")

            sel_indices = None
            if n_inferred > max_frames:
                sel_indices = sorted(set(
                    int(round(i)) for i in np.linspace(0, n_inferred - 1, num=max_frames)
                ))
                print("[info] Uniform sampling: {} inferred -> {} for aggregation".format(
                    n_inferred, len(sel_indices)))
                sampled_preds = [inferred_predictions[idx] for idx in sel_indices]
                accepted_frame_meta = [inferred_frame_meta[idx] for idx in sel_indices]
            else:
                print("[info] Using all {} inferred frames for aggregation".format(n_inferred))
                sampled_preds = list(inferred_predictions)
                accepted_frame_meta = list(inferred_frame_meta)

            multi_init_rows = None
            T_init_used = T_init
            if multi_init_sweep_deg and float(multi_init_sweep_deg) > 0:
                print("[info] Multi-init scan: sweep={}° grid={}".format(
                    multi_init_sweep_deg, multi_init_grid))
                calibrated_T, T_init_used, conf, multi_init_rows = _multi_init_select_best(
                    T_init, accepted_frame_meta, sampled_preds,
                    float(multi_init_sweep_deg), bool(multi_init_grid),
                    wrapper, device, batch_size, img_shape,
                    max_frames, min_agg_frames, T_gt_ref,
                    cv2, torch, xbound, ybound, zbound,
                )
                n_accepted = min(len(sampled_preds), max_frames)
                insufficient_frames = n_accepted < min_agg_frames
                if insufficient_frames:
                    print("[warn] Only {} frames for aggregation (minimum recommended: {}). "
                          "Confidence will be lower.".format(n_accepted, min_agg_frames))
            else:
                aggregator = TemporalCalibrationAggregator(
                    min_frames=min_agg_frames, max_frames=max_frames, method="axis_angle_median",
                )
                for p in sampled_preds:
                    aggregator.add(p)
                n_accepted = aggregator.count
                insufficient_frames = n_accepted < min_agg_frames
                if insufficient_frames:
                    print("[warn] Only {} frames for aggregation (minimum recommended: {}). "
                          "Confidence will be lower.".format(n_accepted, min_agg_frames))
                calibrated_T = aggregator.aggregate()
                conf = aggregator.get_confidence()

            metrics = T_to_calibration_metrics(T_init_used, calibrated_T)

            save_matrix_txt(os.path.join(trip_out, "calibrated_extrinsic.txt"), calibrated_T)
            save_matrix_txt(os.path.join(trip_out, "original_extrinsic.txt"), T_init)

            project_fn, render_fn, err_fn = _lazy_viz_funcs()

            n_proj = max(1, int(round(n_accepted * projection_ratio))) if projection_ratio > 0 else 0
            proj_indices = []
            if n_proj > 0 and n_accepted > 0:
                if n_proj >= n_accepted:
                    proj_indices = list(range(n_accepted))
                else:
                    proj_indices = sorted(set(
                        int(round(i)) for i in np.linspace(0, n_accepted - 1, num=n_proj)
                    ))

            proj_max_w = 1280
            proj_files_for_report = []
            proj_meta_for_report = []

            proj_banner_extra = "Init->Cal={:.3f} deg".format(metrics["rot_geodesic_deg"])
            if T_gt_ref is not None:
                bm = _compute_bias_compensation_metrics(T_init, calibrated_T, T_gt_ref)
                proj_banner_extra = (
                    "Init->Cal={:.3f} | Init->GT={:.3f} Cal->GT={:.3f} | recover {:.0f}%".format(
                        bm["init_cal_delta_deg"], bm["injected_bias_deg"],
                        bm["residual_bias_deg"],
                        bm["compensation_ratio_pct"] if bm["compensation_ratio_pct"] == bm["compensation_ratio_pct"] else 0.0))

            def _gen_one_projection(pidx):
                meta = accepted_frame_meta[pidx]
                stem = meta["stem"]
                ts_display = meta.get("timestamp", "")
                frame_label = "{}  #{}/{}".format(ts_display, pidx + 1, n_accepted) if ts_display else "#{}/{}".format(pidx + 1, n_accepted)
                if meta.get("speed") is not None:
                    frame_label += " v={:.1f}m/s".format(meta["speed"])
                if meta.get("accel") is not None:
                    frame_label += " a={:.1f}m/s2".format(meta["accel"])
                try:
                    im = cv2.imread(meta["img_path"], cv2.IMREAD_COLOR)
                    if im is None:
                        return None
                    im_r = cv2.resize(im, (Wm, Hm), interpolation=cv2.INTER_AREA)
                    pc_raw = load_raw_pointcloud(meta["pc_path"])

                    if project_fn is not None and render_fn is not None:
                        mosaic = render_projection_comparison(
                            im_r, pc_raw[:, :3].astype(np.float32),
                            T_init, calibrated_T, meta["K"],
                            project_fn, render_fn, err_fn,
                            frame_label=frame_label,
                            banner_extra=proj_banner_extra,
                        )
                    else:
                        mosaic = _fallback_projection(
                            im_r, pc_raw[:, :3].astype(np.float32),
                            T_init, calibrated_T, meta["K"], cv2,
                            frame_label=frame_label,
                        )

                    mh, mw = mosaic.shape[:2]
                    if mw > proj_max_w:
                        s = proj_max_w / float(mw)
                        mosaic = cv2.resize(mosaic, (proj_max_w, int(mh * s)), interpolation=cv2.INTER_AREA)

                    fname = "frame_{}_comparison.png".format(stem)
                    out_png = os.path.join(proj_dir, fname)
                    ok = cv2.imwrite(out_png, mosaic, [cv2.IMWRITE_PNG_COMPRESSION, 6])
                    if ok:
                        return (fname, {
                            "frame_idx": pidx + 1,
                            "total_frames": n_accepted,
                            "timestamp": meta.get("timestamp", ""),
                            "speed": meta.get("speed"),
                            "accel": meta.get("accel"),
                        })
                except Exception as exc:
                    print("[warn] projection failed for {}: {}".format(stem, exc))
                return None

            proj_workers = min(8, len(proj_indices)) if proj_indices else 1
            with ThreadPoolExecutor(max_workers=proj_workers) as proj_pool:
                futures = {proj_pool.submit(_gen_one_projection, pidx): pidx
                           for pidx in proj_indices}
                for fut in as_completed(futures):
                    r = fut.result()
                    if r is not None:
                        proj_files_for_report.append(r[0])
                        proj_meta_for_report.append(r[1])

            proj_files_for_report.sort()
            proj_count = len(proj_files_for_report)
            print("[info] Generated {} projection images (ratio={}, workers={})".format(
                proj_count, projection_ratio, proj_workers))

            gt_T = T_gt_ref if T_gt_ref is not None else _try_load_gt_extrinsic(trip_dir, trip_out)
            gt_metrics = None
            bias_metrics = None
            if gt_T is not None:
                gt_metrics = T_to_calibration_metrics(gt_T, calibrated_T)
                bias_metrics = _compute_bias_compensation_metrics(T_init, calibrated_T, gt_T)
                save_matrix_txt(os.path.join(trip_out, "gt_extrinsic.txt"), gt_T)
                print("[info] GT comparison (cal vs true GT): rot_err={:.4f} deg, "
                      "compensation={:.4f} deg ({:.1f}%) shortcut={}".format(
                          gt_metrics["rot_geodesic_deg"],
                          bias_metrics["compensation_deg"],
                          bias_metrics["compensation_ratio_pct"]
                          if bias_metrics["compensation_ratio_pct"] == bias_metrics["compensation_ratio_pct"]
                          else 0.0,
                          bias_metrics["shortcut_risk"]))
                print("[info] Shortcut: {}".format(bias_metrics["shortcut_note"]))

            cam_iae = cam_cfg.get("install_angle_error")
            lidar_iae = lidars.get("install_angle_error")
            is_gt, gt_reason = _check_install_angle_error_is_gt(cam_iae, lidar_iae)
            multi_window_errors = {}
            multi_window_errors_init = {}
            medw_ref = None
            if T_gt_ref is not None:
                medw_ref = T_gt_ref
                print("[info] MEDW reference: true GT ({})".format(gt_lidars_path))
                multi_window_errors = _compute_multi_window_errors(
                    inferred_predictions, T_gt_ref, window_sizes=(50, 100, 200, 400))
                multi_window_errors_init = _compute_multi_window_errors(
                    inferred_predictions, T_init, window_sizes=(50, 100, 200, 400))
                for ws, errs in sorted(multi_window_errors.items()):
                    e_init = multi_window_errors_init.get(ws, {})
                    print("[info] MEDW{} vs GT: rot={:.4f} | vs Init (shortcut): rot={:.4f}".format(
                        ws, errs["rot"], e_init.get("rot", float("nan"))))
                result["gt_source"] = "gt_lidars_cfg"
            elif is_gt:
                medw_ref = T_init
                print("[info] GT condition satisfied: {}".format(gt_reason))
                multi_window_errors = _compute_multi_window_errors(
                    inferred_predictions, T_init, window_sizes=(50, 100, 200, 400))
                for ws, errs in sorted(multi_window_errors.items()):
                    print("[info] MEDW{}: rot={:.4f} roll={:.4f} pitch={:.4f} yaw={:.4f}".format(
                        ws, errs["rot"], errs["roll"], errs["pitch"], errs["yaw"]))
                result["gt_source"] = "install_angle_error"
            else:
                print("[info] Not GT: {}".format(gt_reason))
                result["gt_reason"] = gt_reason

            # Copy trip config directory to output
            _copy_trip_configs(config_dir, trip_out)

            # Copy model folder to output if available
            if trip_dir:
                model_src = os.path.join(trip_dir, "model")
                if os.path.isdir(model_src):
                    model_dst = os.path.join(trip_out, "model")
                    try:
                        if os.path.isdir(model_dst):
                            shutil.rmtree(model_dst)
                        shutil.copytree(model_src, model_dst)
                        print("[info] Copied model dir to {}".format(model_dst))
                    except Exception as exc:
                        print("[warn] Failed to copy model dir: {}".format(exc))

            # Generate lidars_calibrated.cfg with calibrated extrinsic in sensing frame
            _generate_lidars_calibrated_cfg(
                config_dir, trip_out, calibrated_T, cam_cfg, trip_dir=trip_dir)
            iae_cfg_path = os.path.join(trip_out, "configs", "lidars_calibrated.cfg")
            result["iae"] = _read_iae_from_cfg(iae_cfg_path)
            result["orig_iae"] = _read_iae_from_cfg(lidars_cfg)
            model_dir_check = os.path.join(trip_dir, "model") if trip_dir else None
            result["has_model"] = bool(
                model_dir_check
                and os.path.isfile(os.path.join(model_dir_check, "lidars.cfg"))
                and os.path.isfile(os.path.join(model_dir_check, "cameras.cfg"))
            )

            report_path = os.path.join(trip_out, "calibration_report.md")
            _write_trip_report_md(
                report_path, trip_name, n_accepted, extract_cap, epoch,
                T_init, calibrated_T, metrics, qf_stats, conf,
                gt_metrics, proj_files_for_report, proj_meta_for_report,
                is_gt=is_gt, gt_reason=gt_reason,
                multi_window_errors=multi_window_errors,
                multi_window_errors_init=multi_window_errors_init,
                bias_metrics=bias_metrics,
                gt_lidars_path=gt_lidars_path,
                cam_iae=cam_iae, lidar_iae=lidar_iae,
                insufficient_frames=insufficient_frames,
                min_agg_frames=min_agg_frames,
            )

            t_trip_total = time.time() - t_trip_start
            result["status"] = "ok"
            result["frames"] = n_accepted
            result["extract_sec"] = round(t_extract_sec, 1)
            result["infer_sec"] = round(t_infer_sec, 1)
            result["total_sec"] = round(t_trip_total, 1)
            result["rot_delta"] = metrics["rot_geodesic_deg"]
            result["roll_delta"] = metrics.get("roll_delta_deg", float("nan"))
            result["pitch_delta"] = metrics.get("pitch_delta_deg", float("nan"))
            result["yaw_delta"] = metrics.get("yaw_delta_deg", float("nan"))
            result["qf_skip_static"] = qf_stats["skip_static"]
            result["qf_skip_fast"] = qf_stats["skip_fast"]
            result["qf_skip_dark"] = qf_stats["skip_dark"]
            result["qf_skip_air_susp"] = qf_stats.get("skip_air_susp", 0)
            if conf:
                def _std_safe(v):
                    if v is None:
                        return float("nan")
                    return float(v)
                result["roll_std"] = _std_safe(conf.get("roll_std"))
                result["pitch_std"] = _std_safe(conf.get("pitch_std"))
                result["yaw_std"] = _std_safe(conf.get("yaw_std"))
                result["total_std"] = _std_safe(conf.get("total_std"))
            result["is_gt"] = is_gt
            result["gt_reason"] = gt_reason
            result["insufficient_frames"] = insufficient_frames
            if multi_window_errors:
                result["multi_window_errors"] = multi_window_errors
            if multi_window_errors_init:
                result["multi_window_errors_init"] = multi_window_errors_init
            if gt_metrics:
                result["gt_rot_error"] = gt_metrics["rot_geodesic_deg"]
            if bias_metrics:
                result["bias_compensation"] = bias_metrics
                result["injected_bias_deg"] = bias_metrics["injected_bias_deg"]
                result["residual_bias_deg"] = bias_metrics["residual_bias_deg"]
                result["compensation_ratio_pct"] = bias_metrics["compensation_ratio_pct"]
                result["shortcut_risk"] = bias_metrics["shortcut_risk"]

    except Exception as exc:
        result["status"] = "failed"
        result["error"] = "{}: {}".format(type(exc).__name__, exc)
        print("[error] Trip {} failed: {}".format(trip_name, result["error"]))
        traceback.print_exc()
        try:
            with io.open(os.path.join(trip_out, "calibration_report.md"), "w", encoding="utf-8") as handle:
                handle.write("# Calibration FAILED\n\n{}\n".format(result["error"]))
        except Exception:
            pass

    finally:
        if not keep_temp:
            try:
                shutil.rmtree(extract_root, ignore_errors=True)
            except Exception:
                pass
        else:
            dest = os.path.join(trip_out, "_kitti_extract_debug")
            try:
                if os.path.isdir(dest):
                    shutil.rmtree(dest, ignore_errors=True)
                shutil.move(extract_root, dest)
                print("[info] retain temp extract at {}".format(dest))
            except Exception as exc:
                print("[warn] Could not relocate temp extract: {}".format(exc))

    return result


def _copy_trip_configs(config_dir, trip_out):
    """Copy the trip's config directory to the output directory."""
    dst = os.path.join(trip_out, "configs")
    try:
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        shutil.copytree(config_dir, dst)
        print("[info] Copied config dir to {}".format(dst))
    except Exception as exc:
        print("[warn] Failed to copy config dir: {}".format(exc))


def _compute_install_angle_error(T_cal, T_model):
    """
    Compute install_angle_error: RPY difference between calibrated and model extrinsics.

    Replicates the C++ ``evaluate_sensor_extrinsic`` algorithm:
        dR = R_cal @ R_model^T
        axis_angle = AngleAxis(dR).angle * AngleAxis(dR).axis
        result = axis_angle * 180 / pi   (degrees)

    Returns dict with keys ``x`` (roll), ``y`` (pitch), ``z`` (yaw) in degrees.
    """
    ScipyRot = _require_scipy_rot()
    dR = T_cal[:3, :3] @ T_model[:3, :3].T
    rotvec = ScipyRot.from_matrix(dR).as_rotvec()  # radians
    axis_angle_deg = np.degrees(rotvec)
    return {
        "x": float(axis_angle_deg[0]),
        "y": float(axis_angle_deg[1]),
        "z": float(axis_angle_deg[2]),
    }


def _read_iae_from_cfg(cfg_path):
    """Read install_angle_error from a lidars_calibrated.cfg file."""
    try:
        if not os.path.isfile(cfg_path) or ConfigParser is None:
            return {}
        lidars = ConfigParser.parse_lidars_cfg(cfg_path)
        iae = lidars.get("install_angle_error")
        if iae:
            return {
                "roll": float(iae.get("x", 0.0)),
                "pitch": float(iae.get("y", 0.0)),
                "yaw": float(iae.get("z", 0.0)),
            }
    except Exception:
        pass
    return {}


def _generate_lidars_calibrated_cfg(config_dir, trip_out, calibrated_T, cam_cfg,
                                     trip_dir=None):
    """
    Generate lidars_calibrated.cfg with the calibrated lidar-to-sensing extrinsic.

    The calibrated_T is T_lidar_to_cam. To recover the calibrated T_lidar_to_sensing:
        T_cam_to_sensing = build(cam_cfg orientation, position)
        T_lidar_to_sensing_cal = T_cam_to_sensing @ calibrated_T
    Then decompose into quaternion + position for the cfg format.

    If ``trip_dir`` contains ``model/lidars.cfg``, the RPY gap between calibrated
    and model ``sensor_to_lidar`` is computed and written as ``install_angle_error``.
    """
    ScipyRot = _require_scipy_rot()

    src_path = os.path.join(config_dir, "lidars.cfg")
    if not os.path.isfile(src_path):
        print("[warn] lidars.cfg not found, skipping lidars_calibrated.cfg")
        return

    T_cam_to_sensing = np.eye(4, dtype=np.float64)
    T_cam_to_sensing[:3, :3] = ScipyRot.from_quat(cam_cfg["orientation"]).as_matrix()
    T_cam_to_sensing[:3, 3] = cam_cfg["position"]

    T_l2s_cal = T_cam_to_sensing @ calibrated_T

    pos = T_l2s_cal[:3, 3]
    quat = ScipyRot.from_matrix(T_l2s_cal[:3, :3]).as_quat()  # (x, y, z, w)

    with open(src_path, "r") as f:
        content = f.read()

    def _replace_sensor_to_lidar(text, new_pos, new_quat):
        """Replace the main lidar's sensor_to_lidar position and orientation values."""
        pattern = re.compile(
            r'(sensor_to_lidar\s*\{)'
            r'(.*?'
            r'position\s*\{)'
            r'(.*?)'
            r'(\}'  # close position
            r'.*?'
            r'orientation\s*\{)'
            r'(.*?)'
            r'(\})',  # close orientation
            re.DOTALL
        )

        def replacer(m):
            pos_block = (
                "\n      x: {}\n      y: {}\n      z: {}\n    ".format(
                    new_pos[0], new_pos[1], new_pos[2])
            )
            ori_block = (
                "\n      qx: {}\n      qy: {}\n      qz: {}\n      qw: {}\n    ".format(
                    new_quat[0], new_quat[1], new_quat[2], new_quat[3])
            )
            return m.group(1) + m.group(2) + pos_block + m.group(4) + ori_block + m.group(6)

        return pattern.sub(replacer, text, count=1)

    new_content = _replace_sensor_to_lidar(content, pos, quat)

    # Compute install_angle_error vs model extrinsic.
    # Priority: 1) model/lidars.cfg  2) reverse-compute from configs' install_angle_error
    iae = None
    T_l2s_model = None

    model_lid_path = os.path.join(trip_dir, "model", "lidars.cfg") if trip_dir else None
    if model_lid_path and os.path.isfile(model_lid_path):
        try:
            model_lidars = ConfigParser.parse_lidars_cfg(model_lid_path)
            T_l2s_model = np.eye(4, dtype=np.float64)
            T_l2s_model[:3, :3] = ScipyRot.from_quat(model_lidars["orientation"]).as_matrix()
            T_l2s_model[:3, 3] = model_lidars["position"]
            print("[info] Loaded model extrinsic from model/lidars.cfg")
        except Exception as exc:
            print("[warn] Failed to load model/lidars.cfg: {}".format(exc))

    if T_l2s_model is None:
        # Fallback: reverse-compute model from configs' install_angle_error + sensor_to_lidar.
        # IAE = rotvec(R_configs @ R_model^T), so R_model = rotvec_to_mat(IAE)^T @ R_configs.
        try:
            config_lidars = ConfigParser.parse_lidars_cfg(src_path)
            old_iae = config_lidars.get("install_angle_error")
            if old_iae and all(k in old_iae for k in ("x", "y", "z")):
                iae_rad = np.radians([old_iae["x"], old_iae["y"], old_iae["z"]])
                dR = ScipyRot.from_rotvec(iae_rad).as_matrix()
                R_configs = ScipyRot.from_quat(config_lidars["orientation"]).as_matrix()
                R_model = dR.T @ R_configs
                T_l2s_model = np.eye(4, dtype=np.float64)
                T_l2s_model[:3, :3] = R_model
                T_l2s_model[:3, 3] = config_lidars["position"]
                print("[info] Reconstructed model extrinsic from configs install_angle_error")
        except Exception as exc:
            print("[warn] Failed to reconstruct model extrinsic: {}".format(exc))

    if T_l2s_model is not None:
        try:
            iae = _compute_install_angle_error(T_l2s_cal, T_l2s_model)
        except Exception as exc:
            print("[warn] Failed to compute install_angle_error: {}".format(exc))

    def _replace_install_angle_error(text, iae_vals):
        """Replace or insert install_angle_error in the config block."""
        iae_block = (
            "install_angle_error {{\n"
            "    x: {x}\n"
            "    y: {y}\n"
            "    z: {z}\n"
            "  }}".format(**iae_vals)
        )
        pattern = re.compile(
            r'install_angle_error\s*\{[^}]*\}',
            re.DOTALL,
        )
        if pattern.search(text):
            return pattern.sub(iae_block, text, count=1)
        # No existing field — insert before the closing "}" of the config block
        last_brace = text.rfind("}")
        if last_brace > 0:
            return text[:last_brace] + "  " + iae_block + "\n" + text[last_brace:]
        return text

    if iae is not None:
        new_content = _replace_install_angle_error(new_content, iae)

    calib_type_pat = re.compile(r'calib_type\s*:\s*\S+')
    if calib_type_pat.search(new_content):
        new_content = calib_type_pat.sub('calib_type: CALIB_TYPE_ONLINE', new_content, count=1)
    else:
        last_brace = new_content.rfind("}")
        if last_brace > 0:
            new_content = (new_content[:last_brace]
                           + "  calib_type: CALIB_TYPE_ONLINE\n"
                           + new_content[last_brace:])

    is_orig_pat = re.compile(r'is_original\s*:\s*\S+')
    if is_orig_pat.search(new_content):
        new_content = is_orig_pat.sub('is_original: false', new_content, count=1)
    else:
        last_brace = new_content.rfind("}")
        if last_brace > 0:
            new_content = (new_content[:last_brace]
                           + "  is_original: false\n"
                           + new_content[last_brace:])

    dst_path = os.path.join(trip_out, "configs", "lidars_calibrated.cfg")
    if not os.path.isdir(os.path.join(trip_out, "configs")):
        os.makedirs(os.path.join(trip_out, "configs"), exist_ok=True)
    with open(dst_path, "w") as f:
        f.write(new_content)

    print("[info] Generated lidars_calibrated.cfg (sensing frame, calib_type=ONLINE)")
    print("[info]   position: x={:.9f} y={:.9f} z={:.9f}".format(pos[0], pos[1], pos[2]))
    print("[info]   orientation: qx={:.9f} qy={:.9f} qz={:.9f} qw={:.9f}".format(
        quat[0], quat[1], quat[2], quat[3]))
    if iae is not None:
        print("[info]   install_angle_error: roll={:.6f} pitch={:.6f} yaw={:.6f} deg".format(
            iae["x"], iae["y"], iae["z"]))


def _T_to_rpy_xyz(T):
    """Decompose 4x4 transform to (roll, pitch, yaw) in degrees and (x, y, z) in meters."""
    ScipyRot = _require_scipy_rot()
    rpy = ScipyRot.from_matrix(T[:3, :3]).as_euler("xyz", degrees=True)
    return {"roll": float(rpy[0]), "pitch": float(rpy[1]), "yaw": float(rpy[2]),
            "x": float(T[0, 3]), "y": float(T[1, 3]), "z": float(T[2, 3])}


def _write_trip_report_md(
    report_path, trip_name, n_frames, extract_cap, epoch,
    T_init, calibrated_T, metrics, qf_stats, conf,
    gt_metrics, proj_files, proj_meta=None,
    is_gt=False, gt_reason="", multi_window_errors=None,
    multi_window_errors_init=None,
    bias_metrics=None, gt_lidars_path=None,
    cam_iae=None, lidar_iae=None,
    insufficient_frames=False, min_agg_frames=50,
):
    """Write per-trip calibration report as Markdown."""

    T_init_inv = np.linalg.inv(T_init)
    T_cal_inv = np.linalg.inv(calibrated_T)

    orig_l2c = _T_to_rpy_xyz(T_init)
    cal_l2c = _T_to_rpy_xyz(calibrated_T)
    orig_c2l = _T_to_rpy_xyz(T_init_inv)
    cal_c2l = _T_to_rpy_xyz(T_cal_inv)

    lines = []
    lines.append("# Calibration Report: {}".format(trip_name))
    lines.append("")

    lines.append("## Glossary")
    lines.append("")
    lines.append("| Variable | Description |")
    lines.append("| --- | --- |")
    lines.append("| Accepted inference frames | Number of frames that passed all quality filters and were used for inference |")
    lines.append("| Extracted raw frames cap | Maximum raw frames to extract from bags before quality filtering |")
    lines.append("| total_std (deg) | Prediction consistency: L2 norm of per-axis RPY standard deviations across all inferred frames. Lower = more consistent predictions |")
    lines.append("| Conf% | Confidence percentage: `min(1, frames/200) * max(0, 1 - total_std/0.15) * 100`. Combines frame count and prediction stability |")
    lines.append("| Calib Delta / Rot Delta (deg) | Geodesic rotation angle between original and calibrated extrinsic (how much the calibration adjusted) |")
    lines.append("| Roll/Pitch/Yaw Delta (deg) | Per-axis rotation difference between original and calibrated extrinsic |")
    lines.append("| IAE (Install Angle Error) | RPY angular difference between an extrinsic and the factory standard (model/ dir). Orig IAE = original trip extrinsic vs factory; Cal IAE = calibrated extrinsic vs factory |")
    lines.append("| MEDW (Mean Error over Decaying Windows) | Aggregation error vs reference extrinsic (true GT when --gt_lidars_cfg set, else init config) |")
    lines.append("| Injected bias | Geodesic rotation between init T_lidar_to_cam and true GT (systematic offset under test) |")
    lines.append("| Compensation ratio | `(injected - residual) / injected` — fraction of injected bias recovered toward true GT |")
    lines.append("| Shortcut risk | HIGH if model stays near init despite large injected bias, or recovery < 30% |")
    lines.append("| GT Status | Whether install_angle_error fields are within +/-2.5 deg (legacy MEDW gate when no gt_lidars_cfg) |")
    lines.append("")

    lines.append("## Overview")
    lines.append("")
    lines.append("| Item | Value |")
    lines.append("| --- | --- |")
    lines.append("| Accepted inference frames | {} |".format(n_frames))
    lines.append("| Extracted raw frames cap | {} |".format(extract_cap))
    lines.append("| Checkpoint epoch | {} |".format(epoch))
    lines.append("| Total scanned | {} |".format(qf_stats.get("total_scanned", "?")))
    lines.append("| Skipped (static) | {} |".format(qf_stats.get("skip_static", 0)))
    lines.append("| Skipped (too fast) | {} |".format(qf_stats.get("skip_fast", 0)))
    lines.append("| Skipped (high accel) | {} |".format(qf_stats.get("skip_accel", 0)))
    lines.append("| Skipped (dark) | {} |".format(qf_stats.get("skip_dark", 0)))
    lines.append("| Skipped (interval) | {} |".format(qf_stats.get("skip_interval", 0)))
    lines.append("| Skipped (few points) | {} |".format(qf_stats.get("skip_points", 0)))
    lines.append("| Skipped (air susp) | {} |".format(qf_stats.get("skip_air_susp", 0)))
    lines.append("")

    lines.append("## Extrinsic Comparison (RPY + XYZ)")
    lines.append("")
    lines.append("### T_lidar_to_cam (LiDAR -> Camera)")
    lines.append("")
    lines.append("| Component | Original | Calibrated | Delta (Gap) |")
    lines.append("| --- | --- | --- | --- |")
    for key in ["roll", "pitch", "yaw"]:
        delta = cal_l2c[key] - orig_l2c[key]
        lines.append("| {} (deg) | {:.6f} | {:.6f} | {:.6f} |".format(key, orig_l2c[key], cal_l2c[key], delta))
    for key in ["x", "y", "z"]:
        delta = cal_l2c[key] - orig_l2c[key]
        lines.append("| {} (m) | {:.6f} | {:.6f} | {:.6f} |".format(key, orig_l2c[key], cal_l2c[key], delta))
    lines.append("")

    lines.append("### T_cam_to_lidar (Camera -> LiDAR)")
    lines.append("")
    lines.append("| Component | Original | Calibrated | Delta (Gap) |")
    lines.append("| --- | --- | --- | --- |")
    for key in ["roll", "pitch", "yaw"]:
        delta = cal_c2l[key] - orig_c2l[key]
        lines.append("| {} (deg) | {:.6f} | {:.6f} | {:.6f} |".format(key, orig_c2l[key], cal_c2l[key], delta))
    for key in ["x", "y", "z"]:
        delta = cal_c2l[key] - orig_c2l[key]
        lines.append("| {} (m) | {:.6f} | {:.6f} | {:.6f} |".format(key, orig_c2l[key], cal_c2l[key], delta))
    lines.append("")

    lines.append("### Rotation Delta Summary (in LiDAR frame)")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("| --- | --- |")
    lines.append("| Total Rotation (geodesic) | {:.6f} deg |".format(metrics["rot_geodesic_deg"]))
    lines.append("| Roll delta | {:.6f} deg |".format(metrics.get("roll_delta_deg", 0.0)))
    lines.append("| Pitch delta | {:.6f} deg |".format(metrics.get("pitch_delta_deg", 0.0)))
    lines.append("| Yaw delta | {:.6f} deg |".format(metrics.get("yaw_delta_deg", 0.0)))
    lines.append("")

    lines.append("## Extrinsic Matrices (4x4)")
    lines.append("")
    lines.append("### Original T_lidar_to_cam")
    lines.append("")
    lines.append("```")
    lines.append(np.array2string(T_init, precision=8, suppress_small=True))
    lines.append("```")
    lines.append("")
    lines.append("### Calibrated T_lidar_to_cam (MEDW{})".format(n_frames))
    lines.append("")
    lines.append("```")
    lines.append(np.array2string(calibrated_T, precision=8, suppress_small=True))
    lines.append("```")
    lines.append("")
    lines.append("### Original T_cam_to_lidar")
    lines.append("")
    lines.append("```")
    lines.append(np.array2string(T_init_inv, precision=8, suppress_small=True))
    lines.append("```")
    lines.append("")
    lines.append("### Calibrated T_cam_to_lidar")
    lines.append("")
    lines.append("```")
    lines.append(np.array2string(T_cal_inv, precision=8, suppress_small=True))
    lines.append("```")
    lines.append("")

    lines.append("## Confidence")
    lines.append("")
    if conf:
        lines.append("| Metric | Value |")
        lines.append("| --- | --- |")
        def _safe_std(v):
            if v is None or (isinstance(v, float) and v != v):
                return float("nan")
            return float(v)
        lines.append("| roll_std | {:.4f} deg |".format(_safe_std(conf.get("roll_std"))))
        lines.append("| pitch_std | {:.4f} deg |".format(_safe_std(conf.get("pitch_std"))))
        lines.append("| yaw_std | {:.4f} deg |".format(_safe_std(conf.get("yaw_std"))))
        lines.append("| total_std | {:.4f} deg |".format(_safe_std(conf.get("total_std"))))
        lines.append("| n_frames | {} |".format(conf.get("n_frames", n_frames)))
        ts = _safe_std(conf.get("total_std"))
        if ts != ts:
            ts = 999.0
        frame_f = min(1.0, n_frames / 200.0) if n_frames > 0 else 0
        stab_f = max(0.0, 1.0 - ts / 0.15)
        conf_pct = round(frame_f * stab_f * 100, 1)
        if insufficient_frames:
            lines.append("")
            lines.append("Confidence: **LOW** ({:.0f}%, insufficient frames)".format(conf_pct))
            lines.append("")
            lines.append("> **WARNING**: Only {} frames available (recommended minimum: {}). "
                         "Calibration is based on all available frames but **confidence is LOW**. "
                         "The calibrated extrinsic may not be reliable. "
                         "Consider using more bag data or relaxing quality filters.".format(
                             n_frames, min_agg_frames))
        elif ts < 0.05:
            lines.append("")
            lines.append("Confidence: **HIGH** ({:.0f}%, total_std < 0.05 deg)".format(conf_pct))
        elif ts < 0.15:
            lines.append("")
            lines.append("Confidence: **MEDIUM** ({:.0f}%, 0.05 <= total_std < 0.15 deg)".format(conf_pct))
        else:
            lines.append("")
            lines.append("Confidence: **LOW** ({:.0f}%, total_std >= 0.15 deg)".format(conf_pct))
    else:
        lines.append("(insufficient frames)")
    lines.append("")

    lines.append("## GT Qualification (install_angle_error)")
    lines.append("")
    if cam_iae:
        lines.append("- traffic_2 install_angle_error: roll={:.3f} pitch={:.3f} yaw={:.3f} deg".format(
            cam_iae["x"], cam_iae["y"], cam_iae["z"]))
    else:
        lines.append("- traffic_2 install_angle_error: **NOT FOUND**")
    if lidar_iae:
        lines.append("- main LiDAR install_angle_error: roll={:.3f} pitch={:.3f} yaw={:.3f} deg".format(
            lidar_iae["x"], lidar_iae["y"], lidar_iae["z"]))
    else:
        lines.append("- main LiDAR install_angle_error: **NOT FOUND**")
    lines.append("- **GT Status**: {} ({})".format("YES" if is_gt else "NO", gt_reason))
    lines.append("")

    if bias_metrics:
        lines.append("## Systematic Bias & Shortcut Analysis")
        lines.append("")
        if gt_lidars_path:
            lines.append("True GT source: `{}`".format(gt_lidars_path))
            lines.append("")
        lines.append("| Metric | Value |")
        lines.append("| --- | --- |")
        lines.append("| Injected bias (Init vs true GT) | {:.4f} deg |".format(
            bias_metrics["injected_bias_deg"]))
        lines.append("| Residual bias (Cal vs true GT) | {:.4f} deg |".format(
            bias_metrics["residual_bias_deg"]))
        lines.append("| Compensation (improvement) | {:.4f} deg |".format(
            bias_metrics["compensation_deg"]))
        cr = bias_metrics["compensation_ratio_pct"]
        lines.append("| Compensation ratio | {} |".format(
            "{:.1f}%".format(cr) if cr == cr else "N/A"))
        lines.append("| Init→Cal adjustment | {:.4f} deg |".format(
            bias_metrics["init_cal_delta_deg"]))
        lines.append("| Moved toward true GT | {} |".format(
            "YES" if bias_metrics["moved_toward_gt"] else "NO"))
        lines.append("| **Shortcut risk** | **{}** |".format(bias_metrics["shortcut_risk"]))
        lines.append("")
        lines.append("> {}".format(bias_metrics["shortcut_note"]))
        lines.append("")
        lines.append("Per-axis vs true GT (deg):")
        lines.append("")
        lines.append("| Stage | Roll | Pitch | Yaw |")
        lines.append("| --- | ---: | ---: | ---: |")
        lines.append("| Init vs GT | {:+.4f} | {:+.4f} | {:+.4f} |".format(
            bias_metrics["init_vs_gt_roll_deg"],
            bias_metrics["init_vs_gt_pitch_deg"],
            bias_metrics["init_vs_gt_yaw_deg"]))
        lines.append("| Cal vs GT | {:+.4f} | {:+.4f} | {:+.4f} |".format(
            bias_metrics["cal_vs_gt_roll_deg"],
            bias_metrics["cal_vs_gt_pitch_deg"],
            bias_metrics["cal_vs_gt_yaw_deg"]))
        lines.append("")

    if multi_window_errors:
        ref_label = "true GT" if gt_lidars_path else "init extrinsic"
        lines.append("## Calibration Accuracy (MEDW vs {})".format(ref_label))
        lines.append("")
        lines.append("| Window | Total Rot (deg) | Roll (deg) | Pitch (deg) | Yaw (deg) |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for ws in sorted(multi_window_errors.keys()):
            e = multi_window_errors[ws]
            lines.append("| MEDW{} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |".format(
                ws, e["rot"], e["roll"], e["pitch"], e["yaw"]))
        lines.append("")
        if multi_window_errors_init:
            lines.append("### MEDW vs Init (shortcut indicator — should be higher if model tracks init)")
            lines.append("")
            lines.append("| Window | Total Rot (deg) | Roll | Pitch | Yaw |")
            lines.append("| --- | ---: | ---: | ---: | ---: |")
            for ws in sorted(multi_window_errors_init.keys()):
                e = multi_window_errors_init[ws]
                lines.append("| MEDW{} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |".format(
                    ws, e["rot"], e["roll"], e["pitch"], e["yaw"]))
            lines.append("")

    lines.append("## Output Files")
    lines.append("")
    lines.append("- [calibrated_extrinsic.txt](calibrated_extrinsic.txt)")
    lines.append("- [original_extrinsic.txt](original_extrinsic.txt)")
    if gt_lidars_path:
        lines.append("- [gt_extrinsic.txt](gt_extrinsic.txt)")
    lines.append("- [calibration.log](calibration.log)")
    lines.append("- [projections/](projections/) ({} images)".format(len(proj_files)))
    lines.append("")

    if proj_files:
        lines.append("## Projection Comparison Gallery")
        lines.append("")
        lines.append("Left = Original extrinsic | Right = Calibrated extrinsic")
        lines.append("")
        lines.append("| Frame | Timestamp | Speed / Accel | Projection |")
        lines.append("| --- | --- | --- | --- |")
        for i in range(len(proj_files)):
            pm = proj_meta[i] if proj_meta and i < len(proj_meta) else {}
            frame_col = "#{}/{}".format(pm.get("frame_idx", i + 1), pm.get("total_frames", len(proj_files)))
            ts_col = pm.get("timestamp", "-")
            dyn_parts = []
            if pm.get("speed") is not None:
                dyn_parts.append("{:.1f} m/s".format(pm["speed"]))
            if pm.get("accel") is not None:
                dyn_parts.append("{:.2f} m/s2".format(pm["accel"]))
            dyn_col = " / ".join(dyn_parts) if dyn_parts else "-"
            lines.append("| {} | {} | {} | ![](projections/{}) |".format(
                frame_col, ts_col, dyn_col, proj_files[i]))
        lines.append("")
        lines.append("Total {} projection images.".format(len(proj_files)))
        lines.append("")

    with io.open(report_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


# =============================================================================
# Section: Scheduling (sequential vs multi-GPU subprocess fan-out)
# =============================================================================
def resolve_parallel_plan(parallel_flag, num_trips):
    """
    Decide how many CUDA workers to use.

    Returns:
        int: GPU worker count (0 means in-process sequential).
    """
    torch = _require_torch()
    cuda_n = torch.cuda.device_count() if torch.cuda.is_available() else 0

    if parallel_flag == 0:
        return 0

    if parallel_flag == -1:
        if cuda_n <= 1 or num_trips <= 1:
            return 0
        return max(1, min(cuda_n, num_trips, 8))

    n = int(parallel_flag)
    if n <= 0:
        return 0

    if cuda_n == 0:
        print("[warn] --parallel requested but CUDA unavailable — sequential CPU/GPU fallback.")
        return 0

    return max(1, min(n, cuda_n, num_trips))


def dispatch_trips(kwargs_base, trips_payload, parallel_workers, job_tmp, script_path, python_exe):
    """
    Run trips either sequentially or via subprocess isolation per GPU lane.

    Args:
        kwargs_base (dict): Common keyword args for ``calibrate_trip`` excluding per-trip pieces.
        trips_payload (list[dict]): Each dict includes keys trip_name,trip_dir,bag_paths,config_dir.
        parallel_workers (int): From ``resolve_parallel_plan``.
        job_tmp (str): Directory for ephemeral JSON descriptors.
        script_path (str): Path to *this* file for subprocess respawn.
        python_exe (str): Interpreter path.

    Returns:
        list[dict]: Calibration results appended in trip order submission (completion order may differ — we resort).
    """

    results = []

    def run_one(payload):
        job = {"trip_name": payload["trip_name"], "kwargs": kwargs_base}
        if payload.get("trip_dir"):
            job["trip_dir"] = payload["trip_dir"]
        else:
            job["trip_dir"] = None

        job["bag_paths"] = payload.get("bag_paths")
        job["config_dir"] = payload.get("config_dir")

        gpu_id = payload.get("gpu_id", 0)

        if parallel_workers == 0:
            merged = dict(kwargs_base)
            return calibrate_trip(
                trip_name=payload["trip_name"],
                trip_dir=payload.get("trip_dir"),
                bag_paths=payload.get("bag_paths"),
                config_dir=payload.get("config_dir"),
                **merged
            )

        json_job = dict(job)
        kw = dict(kwargs_base)
        kw["img_shape"] = [int(kwargs_base["img_shape"][0]), int(kwargs_base["img_shape"][1])]
        json_job["kwargs"] = kw
        job_path = write_trip_job(job_tmp, json_job)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        cmd = [
            python_exe,
            script_path,
            "--from-job-json",
            job_path,
        ]
        timeout = kwargs_base.get("trip_timeout_sec")
        proc = subprocess.run(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=None if timeout is None or timeout <= 0 else int(timeout),
        )
        txt = proc.stdout.decode("utf-8", errors="replace")
        log_dir = os.path.join(kwargs_base["output_dir"], payload["trip_name"])
        os.makedirs(log_dir, exist_ok=True)
        with io.open(
            os.path.join(log_dir, "worker_tail.log"),
            "w",
            encoding="utf-8",
        ) as tail:
            tail.write(txt)
        if proc.returncode != 0:
            return {
                "trip": payload["trip_name"],
                "status": "failed",
                "frames": 0,
                "rot_delta": float("nan"),
                "trans_delta": float("nan"),
                "roll_std": float("nan"),
                "pitch_std": float("nan"),
                "yaw_std": float("nan"),
                "total_std": float("nan"),
                "error": "subprocess rc={} (see worker_tail.log)".format(proc.returncode),
            }
        sidecar = os.path.join(
            kwargs_base["output_dir"], payload["trip_name"], "result_sidecar.json"
        )
        if os.path.isfile(sidecar):
            with io.open(sidecar, "r", encoding="utf-8") as handle:
                return json.load(handle)
        return {
            "trip": payload["trip_name"],
            "status": "unknown",
            "error": "missing result_sidecar.json",
        }

    if parallel_workers == 0:
        for i, payload in enumerate(trips_payload):
            payload = dict(payload)
            payload["gpu_id"] = 0
            results.append(run_one(payload))
        return results

    # Fan-out with a thread pool to cap concurrent subprocesses at parallel_workers
    with ThreadPoolExecutor(max_workers=parallel_workers) as pool:
        futures = {}
        for idx, payload in enumerate(trips_payload):
            p = dict(payload)
            p["gpu_id"] = idx % max(1, parallel_workers)
            futures[pool.submit(run_one, p)] = p["trip_name"]

        for fut in as_completed(futures):
            name = futures[fut]
            try:
                res = fut.result()
            except subprocess.TimeoutExpired:
                res = {
                    "trip": name,
                    "status": "failed",
                    "error": "subprocess timeout",
                }
            except Exception as exc:
                res = {
                    "trip": name,
                    "status": "failed",
                    "error": str(exc),
                }
            results.append(res)

    # Re-order to original trip list for stable markdown
    by_name = {r.get("trip"): r for r in results}
    ordered = []
    for payload in trips_payload:
        t = payload["trip_name"]
        ordered.append(by_name.get(t, {"trip": t, "status": "missing", "error": "no result"}))
    return ordered


# =============================================================================
# Section: CLI
# =============================================================================
def build_arg_parser():
    p = argparse.ArgumentParser(
        description="BEVCalib production calibration from rosbags via BEVCalibDatasetPreparer + inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input_file", type=str, default=None, help="Path to bag_list.txt or trips.txt")
    p.add_argument(
        "--input_format",
        type=str,
        choices=["auto", "bag_list", "trips", "remote"],
        default="auto",
        help="Input interpretation. 'remote' downloads trips via drfile before calibration.",
    )
    p.add_argument(
        "--trips_base",
        type=str,
        default="/mnt/drtraining/user/dahailu/data/bevcalib/trips",
        help="Root directory when using trips/remote mode.",
    )
    p.add_argument(
        "--max_bag_groups",
        type=int,
        default=20,
        help="Max time-slot groups to download in remote mode (default: 20).",
    )
    p.add_argument(
        "--bags_base",
        type=str,
        default=".",
        help="Base path when bag_list entries are relative.",
    )
    p.add_argument(
        "--config_dir",
        type=str,
        default=None,
        help="Explicit directory with cameras.cfg + lidars.cfg (required for bag_list mode).",
    )
    p.add_argument(
        "--gt_lidars_cfg",
        type=str,
        default=None,
        help="Path to true-GT lidars.cfg for bias/shortcut metrics and MEDW reference. "
             "If omitted, auto-detects lidars_bk.cfg in config_dir when present.",
    )
    p.add_argument(
        "--inject_lidar_rpy_deg",
        type=str,
        default=None,
        help="Extra sensor_to_lidar rotation injected on top of configs/lidars.cfg before "
             "inference, format 'roll,pitch,yaw' in degrees (e.g. '-0.7,0.35,0' or '2,0,0' for stress test).",
    )
    p.add_argument("--ckpt_path", type=str, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--max_frames", type=int, default=400, help="Aggregator max frames window.")
    p.add_argument(
        "--min_agg_frames",
        type=int,
        default=50,
        help="Informational minimum for aggregator readiness (still aggregates if fewer).",
    )
    p.add_argument("--camera_name", type=str, default="traffic_2")
    p.add_argument(
        "--sample_strategy",
        type=str,
        choices=["time", "distance", "adaptive", "none"],
        default="time",
    )
    p.add_argument("--sample_interval", type=float, default=1.0)
    p.add_argument(
        "--parallel",
        type=int,
        default=0,
        help="0 = in-process sequential; -1 = auto GPU fan-out; N = up to N concurrent subprocesses.",
    )
    p.add_argument("--batch_size", type=int, default=24)
    p.add_argument("--img_h", type=int, default=360)
    p.add_argument("--img_w", type=int, default=640)
    p.add_argument(
        "--projection_ratio",
        type=float,
        default=0.1,
        help="Fraction of accepted frames to generate projections (0.1 = 10%%, 0 = disabled).",
    )
    p.add_argument("--keep_temp", action="store_true", help="Keep KITTI extract under trip folder.")
    p.add_argument("--force_config", action="store_true", help="Forward --force-config to preparer.")
    p.add_argument(
        "--preparer_num_workers",
        type=int,
        default=16,
        help="Thread workers for rosbag decoding. Auto-capped by bag count.",
    )
    p.add_argument(
        "--extract_buffer_multiplier",
        type=float,
        default=1.5,
        help="Extractor pulls up to max_frames * multiplier raw frames before sampling. "
             "Capped at 1.3x in streaming mode. Default 1.5.",
    )
    p.add_argument(
        "--infer_cap_multiplier",
        type=float,
        default=1.25,
        help="Stop inferring after max_frames * this multiplier (streaming). Default 1.25 (~250 for MEDW200).",
    )
    p.add_argument(
        "--multi_init_sweep_deg",
        type=float,
        default=0.0,
        help="Multi-init workaround: sweep ±deg on each RPY axis (0=off). "
             "Re-infers with perturbed T_init and picks best vs GT or confidence.",
    )
    p.add_argument(
        "--multi_init_grid",
        action="store_true",
        default=False,
        help="Use coarse RPY grid instead of axis-only sweep (slower, 124 candidates).",
    )
    p.add_argument(
        "--incremental_extract",
        type=lambda s: str(s).lower() in ("1", "true", "yes", "on"),
        default=True,
        help="Release local bags time-slot-by-time-slot (default true). Set false to stage all bags at once.",
    )
    p.add_argument(
        "--trip_timeout_sec",
        type=int,
        default=0,
        help="Hard timeout per trip subprocess (0 disables).",
    )
    p.add_argument("--min_speed_kmh", type=float, default=5.0,
                   help="Skip frames below this speed (km/h). Default 5.")
    p.add_argument("--max_speed_kmh", type=float, default=120.0,
                   help="Skip frames above this speed (km/h). Default 120.")
    p.add_argument("--min_accel", type=float, default=-1.0,
                   help="Skip frames with accel below this (m/s^2). Default -1.0.")
    p.add_argument("--max_accel", type=float, default=2.0,
                   help="Skip frames with accel above this (m/s^2). Default 2.0.")
    p.add_argument("--min_brightness", type=int, default=25,
                   help="Skip frames with mean brightness below this. Default 25.")
    p.add_argument("--cleanup_cache", action="store_true", default=False,
                   help="Delete cached trip data after calibration (remote mode only).")
    p.add_argument("--feishu_webhook", type=str, default="",
                   help="Feishu webhook URL to push results after calibration.")
    p.add_argument("--filter_air_suspension", action="store_true", default=False,
                   help="Filter out frames where air suspension is not in NORMAL gear. "
                        "Reads /canbus/car_state from bags to detect suspension level.")
    p.add_argument(
        "--from-job-json",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--analyze_dir",
        type=str,
        default=None,
        help="Skip calibration; read all model subdirs under this path and "
             "generate a cross-model generalization ranking report.",
    )
    return p


def _generate_cross_model_report(analyze_dir):
    """Scan all model subdirs, read result_sidecar.json, produce ranking report."""
    model_data = {}
    for model_dir_name in sorted(os.listdir(analyze_dir)):
        model_path = os.path.join(analyze_dir, model_dir_name)
        if not os.path.isdir(model_path) or model_dir_name.startswith(("_", ".")):
            continue
        trips = {}
        for trip_name in sorted(os.listdir(model_path)):
            sidecar = os.path.join(model_path, trip_name, "result_sidecar.json")
            if not os.path.isfile(sidecar):
                continue
            try:
                with io.open(sidecar, "r", encoding="utf-8") as f:
                    data = json.load(f)
                trips[trip_name] = data
            except Exception:
                continue
        if trips:
            model_data[model_dir_name] = trips

    if not model_data:
        print("[warn] No model data found in {}".format(analyze_dir))
        return

    all_trips = sorted(set(t for trips in model_data.values() for t in trips))
    trip_short = {t: t.split("_")[0] for t in all_trips}

    lines = []
    lines.append("# BAG 泛化评估跨模型汇总报告\n")
    lines.append("评估目录: `{}`\n".format(analyze_dir))
    lines.append("模型数: {}  行程数: {}\n".format(len(model_data), len(all_trips)))
    lines.append("行程: {}\n".format(", ".join(trip_short[t] for t in all_trips)))

    def _medw200(data):
        v = data.get("multi_window_errors", {}).get("200", {}).get("rot")
        return float(v) if v is not None else float("nan")

    def _medw200_rpy(data):
        w = data.get("multi_window_errors", {}).get("200", {})
        def _s(k):
            v = w.get(k)
            return float(v) if v is not None else float("nan")
        return _s("roll"), _s("pitch"), _s("yaw")

    # Overall MEDW200 ranking
    model_avg_medw = []
    for model, trips in model_data.items():
        medws = [_medw200(d) for d in trips.values() if _medw200(d) == _medw200(d)]
        if medws:
            model_avg_medw.append((model, float(np.mean(medws)), float(np.std(medws)),
                                   float(np.min(medws)), float(np.max(medws)), len(medws)))
    model_avg_medw.sort(key=lambda x: x[1])

    lines.append("\n## 一、MEDW200 总排名 (跨行程均值, 越低越好)\n")
    header = "| 排名 | 模型 | Avg MEDW200 | Std | Min | Max | #Trips |"
    lines.append(header)
    lines.append("| ---: | --- | ---: | ---: | ---: | ---: | ---: |")
    for rank, (model, avg, std, mn, mx, n) in enumerate(model_avg_medw, 1):
        lines.append("| {} | {} | {:.4f}° | {:.4f} | {:.4f} | {:.4f} | {} |".format(
            rank, model, avg, std, mn, mx, n))

    # Per-trip MEDW200 comparison
    lines.append("\n## 二、Per-Trip MEDW200 对比\n")
    trip_headers = " | ".join(trip_short[t] for t in all_trips)
    lines.append("| 模型 | {} | Avg |".format(trip_headers))
    lines.append("| --- | {} | ---: |".format(" | ".join(["---:"] * len(all_trips))))
    for model, avg, _, _, _, _ in model_avg_medw:
        trips = model_data[model]
        cells = []
        for t in all_trips:
            if t in trips:
                m = _medw200(trips[t])
                gs = trips[t].get("gt_source", "?")
                flag = "*" if gs == "install_angle_error" else ""
                cells.append("{:.4f}{}".format(m, flag) if m == m else "N/A")
            else:
                cells.append("-")
        lines.append("| {} | {} | {:.4f} |".format(model, " | ".join(cells), avg))
    lines.append("\n> \\* = MEDW参考基准为init外参(非GT), 指标不可靠\n")

    # Per-trip RPY breakdown
    lines.append("\n## 三、Per-Trip RPY 分量 (MEDW200)\n")
    for t in all_trips:
        lines.append("\n### {}\n".format(t))
        lines.append("| 模型 | MEDW200 | Roll | Pitch | Yaw | GT源 | Conf% |")
        lines.append("| --- | ---: | ---: | ---: | ---: | --- | ---: |")
        trip_models = []
        for model, avg, _, _, _, _ in model_avg_medw:
            if t in model_data[model]:
                d = model_data[model][t]
                trip_models.append((model, d))
        trip_models.sort(key=lambda x: _medw200(x[1]))
        for model, d in trip_models:
            m = _medw200(d)
            r, p, y = _medw200_rpy(d)
            gs = d.get("gt_source", "?")[:10]
            conf = d.get("total_std")
            if conf is None or (isinstance(conf, float) and conf != conf):
                conf_pct = "?"
            else:
                _fr = d.get("frames") or 0
                conf_pct = "{}%".format(int(min(1, _fr / 200) *
                                            max(0, 1 - conf / 0.15) * 100))
            lines.append("| {} | {:.4f}° | {:.4f} | {:.4f} | {:.4f} | {} | {} |".format(
                model, m, r, p, y, gs, conf_pct))

    # Scenario comparison (baseline vs shortcut vs inject)
    scenario_suffixes = [("_inject_small", "inject_small"),
                         ("_shortcut", "shortcut"),
                         ("_baseline", "baseline")]
    scenarios = {}
    for model in model_data:
        for suffix, scenario in scenario_suffixes:
            if model.endswith(suffix):
                base_model = model[:-len(suffix)]
                scenarios.setdefault(base_model, {})[scenario] = model
                break

    if scenarios:
        lines.append("\n## 四、场景对比 (baseline vs shortcut vs inject_small)\n")
        lines.append("| 模型 | Baseline | Shortcut | Inject Small | Recovery% | Shortcut Risk |")
        lines.append("| --- | ---: | ---: | ---: | ---: | --- |")
        for base_model in sorted(scenarios.keys()):
            sc = scenarios[base_model]
            bl_medw = inj_medw = sc_medw = float("nan")
            recovery = shortcut_risk = "N/A"

            for scenario, full_name in sc.items():
                trips = model_data[full_name]
                medws = [_medw200(d) for d in trips.values() if _medw200(d) == _medw200(d)]
                avg = float(np.mean(medws)) if medws else float("nan")
                if scenario == "baseline":
                    bl_medw = avg
                elif scenario == "shortcut":
                    sc_medw = avg
                    risks = [d.get("shortcut_risk") for d in trips.values()
                             if d.get("shortcut_risk")]
                    shortcut_risk = "/".join(sorted(set(risks))) if risks else "N/A"
                elif scenario == "inject_small":
                    inj_medw = avg
                    recs = []
                    for d in trips.values():
                        cr = d.get("compensation_ratio_pct")
                        if cr is not None and isinstance(cr, (int, float)) and cr == cr:
                            recs.append(float(cr))
                    if recs:
                        recovery = "{:.1f}%".format(np.mean(recs))

            bl_s = "{:.4f}°".format(bl_medw) if bl_medw == bl_medw else "-"
            sc_s = "{:.4f}°".format(sc_medw) if sc_medw == sc_medw else "-"
            inj_s = "{:.4f}°".format(inj_medw) if inj_medw == inj_medw else "-"
            lines.append("| {} | {} | {} | {} | {} | {} |".format(
                base_model, bl_s, sc_s, inj_s, recovery, shortcut_risk))

    # Cross-vehicle consistency
    lines.append("\n## 五、跨车型一致性\n")
    for model, avg, std, mn, mx, n in model_avg_medw[:10]:
        if n < 2:
            continue
        cv = std / avg * 100 if avg > 0 else 0
        verdict = "优秀" if cv < 10 else ("良好" if cv < 20 else "需改进")
        lines.append("- **{}**: Avg={:.4f}° Std={:.4f} CV={:.1f}% → {}".format(
            model, avg, std, cv, verdict))

    report_path = os.path.join(analyze_dir, "CROSS_MODEL_REPORT.md")
    with io.open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("[info] Cross-model report: {}".format(report_path))
    print("[info] Top 5 models by MEDW200:")
    for rank, (model, avg, _, _, _, _) in enumerate(model_avg_medw[:5], 1):
        print("  {}. {} = {:.4f}°".format(rank, model, avg))


def _sanitize_for_json(obj):
    """Coerce NaN/inf and numpy scalars for strict JSON (Feishu-friendly pipelines)."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            out[str(k)] = _sanitize_for_json(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        x = float(obj)
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def _load_yaml_config(yaml_path):
    """Load a YAML config and return a flat dict of CLI-style arguments."""
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    flat = {}
    for k, v in cfg.items():
        flat[k] = v
    return flat


def _cli_explicit_args(argv):
    """Return set of arg names explicitly present on the command line."""
    explicit = set()
    for tok in argv:
        if tok.startswith("--"):
            name = tok.split("=", 1)[0].lstrip("-").replace("-", "_")
            explicit.add(name)
    return explicit


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    has_config_flag = "--config" in sys.argv
    has_analyze_flag = "--analyze_dir" in sys.argv
    cli_explicit = _cli_explicit_args(sys.argv[1:])

    parser = build_arg_parser()
    if has_config_flag or has_analyze_flag:
        for action in parser._actions:
            if action.dest in ("input_file", "ckpt_path", "output_dir"):
                action.required = False
    parser.add_argument("--config", type=str, default=None,
                        help="YAML config file. CLI args override YAML values.")
    args = parser.parse_args()

    if args.config:
        yaml_cfg = _load_yaml_config(args.config)
        for k, v in yaml_cfg.items():
            k_attr = k.replace("-", "_")
            if k_attr not in cli_explicit:
                setattr(args, k_attr, v)

    analyze_dir = getattr(args, "analyze_dir", None)
    if analyze_dir:
        _generate_cross_model_report(analyze_dir)
        return

    if args.from_job_json:
        with io.open(args.from_job_json, "r", encoding="utf-8") as handle:
            job = json.load(handle)
        res = run_trip_job_from_json(args.from_job_json)
        out_dir = job["kwargs"]["output_dir"]
        trip = job["trip_name"]
        os.makedirs(os.path.join(out_dir, trip), exist_ok=True)
        sidecar = os.path.join(out_dir, trip, "result_sidecar.json")
        with io.open(sidecar, "w", encoding="utf-8") as handle:
            json.dump(_sanitize_for_json(res), handle, indent=2, allow_nan=False)
        return

    for required_arg in ("input_file", "ckpt_path", "output_dir"):
        if not getattr(args, required_arg, None):
            raise SystemExit("--{} is required (via CLI or YAML config)".format(required_arg))

    input_format = args.input_format
    if input_format == "auto":
        input_format = detect_input_format(args.input_file)

    trips_payload = []

    if input_format == "bag_list":
        tname, bags = parse_bag_list_file(args.input_file, args.bags_base)
        if not args.config_dir:
            raise SystemExit(
                "bag_list mode requires --config_dir pointing to cameras.cfg and lidars.cfg."
            )
        cfg_dir = os.path.abspath(args.config_dir)
        trips_payload.append(
            {
                "trip_name": tname,
                "trip_dir": None,
                "bag_paths": bags,
                "config_dir": cfg_dir,
            }
        )

    if input_format in ("trips", "remote"):
        trip_names = parse_trips_file(args.input_file)
        for t in trip_names:
            if input_format == "remote":
                tdir = download_remote_trip(t, args.trips_base,
                                           max_bag_groups=args.max_bag_groups)
            else:
                tdir = find_trip_directory(args.trips_base, t)
            trips_payload.append(
                {
                    "trip_name": t,
                    "trip_dir": tdir,
                    "bag_paths": None,
                    "config_dir": None,
                }
            )

    if not trips_payload:
        raise SystemExit("No trips to process — check --input_file / --input_format.")

    os.makedirs(args.output_dir, exist_ok=True)

    torch = _require_torch()
    parallel_plan = resolve_parallel_plan(args.parallel, len(trips_payload))

    kwargs_base = {
        "output_dir": os.path.abspath(args.output_dir),
        "ckpt_path": os.path.abspath(args.ckpt_path),
        "max_frames": int(args.max_frames),
        "min_agg_frames": int(args.min_agg_frames),
        "extract_buffer_multiplier": float(args.extract_buffer_multiplier),
        "camera_name": args.camera_name,
        "sample_strategy": args.sample_strategy,
        "sample_interval": float(args.sample_interval),
        "device": "cpu",
        "batch_size": int(args.batch_size),
        "img_shape": (int(args.img_h), int(args.img_w)),
        "projection_ratio": float(args.projection_ratio),
        "keep_temp": bool(args.keep_temp),
        "force_config": bool(args.force_config),
        "preparer_num_workers": int(args.preparer_num_workers),
        "trip_timeout_sec": int(args.trip_timeout_sec) if args.trip_timeout_sec else None,
        "min_speed_kmh": float(args.min_speed_kmh),
        "max_speed_kmh": float(args.max_speed_kmh),
        "min_accel": float(args.min_accel),
        "max_accel": float(args.max_accel),
        "min_brightness": int(args.min_brightness),
        "filter_air_suspension": bool(getattr(args, "filter_air_suspension", False)),
        "input_format": args.input_format,
        "initial_bag_groups": int(getattr(args, "initial_bag_groups", 3)),
        "gt_lidars_cfg": getattr(args, "gt_lidars_cfg", None),
        "inject_lidar_rpy_deg": getattr(args, "inject_lidar_rpy_deg", None),
        "incremental_extract": getattr(args, "incremental_extract", True),
        "infer_cap_multiplier": float(getattr(args, "infer_cap_multiplier", 1.25)),
        "multi_init_sweep_deg": float(getattr(args, "multi_init_sweep_deg", 0.0)),
        "multi_init_grid": bool(getattr(args, "multi_init_grid", False)),
    }

    if torch.cuda.is_available():
        kwargs_base["device"] = "cuda:0"

    job_tmp = os.path.join(args.output_dir, "_job_payloads")
    os.makedirs(job_tmp, exist_ok=True)
    script_path = os.path.abspath(__file__)
    python_exe = sys.executable

    print(
        "[info] trips={} parallel_workers={} device_template={}".format(
            len(trips_payload), parallel_plan, kwargs_base["device"]
        )
    )

    if parallel_plan > 0 and _ACTIVE_DOWNLOADERS:
        print("[info] Parallel mode: stopping parent downloaders — subprocesses will stream independently")
        for dl_trip, dl in list(_ACTIVE_DOWNLOADERS.items()):
            dl.stop()
        _ACTIVE_DOWNLOADERS.clear()

    results = dispatch_trips(
        kwargs_base,
        trips_payload,
        parallel_plan,
        job_tmp,
        script_path,
        python_exe,
    )

    ok = sum(1 for r in results if r.get("status") == "ok")
    failures = [r for r in results if r.get("status") != "ok"]
    rot_ok = []
    for r in results:
        if r.get("status") != "ok":
            continue
        _rd_raw = r.get("rot_delta")
        rd = float(_rd_raw) if _rd_raw is not None else float("nan")
        if rd == rd:
            rot_ok.append(rd)
    total_extract = sum(r.get("extract_sec", 0) or 0 for r in results if r.get("status") == "ok")
    total_infer = sum(r.get("infer_sec", 0) or 0 for r in results if r.get("status") == "ok")
    total_all = sum(r.get("total_sec", 0) or 0 for r in results if r.get("status") == "ok")
    global_stats = {
        "trip_total": len(results),
        "trip_ok": ok,
        "trip_failed": len(failures),
        "mean_rot_delta_deg": float(np.mean(rot_ok)) if rot_ok else float("nan"),
        "std_rot_delta_deg": float(np.std(rot_ok)) if rot_ok else float("nan"),
        "total_extract_sec": round(total_extract, 1),
        "total_infer_sec": round(total_infer, 1),
        "total_time_sec": round(total_all, 1),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    images_dir = os.path.join(args.output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    chart_files = generate_summary_rpy_charts(results, images_dir)

    rpy_stats_path = os.path.join(images_dir, "_rpy_stats.md")
    rpy_stats_content = ""
    if os.path.isfile(rpy_stats_path):
        with io.open(rpy_stats_path, "r", encoding="utf-8") as f:
            rpy_stats_content = f.read()

    report_md = os.path.join(args.output_dir, "SUMMARY_REPORT.md")
    with io.open(report_md, "w", encoding="utf-8") as handle:
        handle.write(build_markdown_summary(results, global_stats))

        if chart_files:
            handle.write("\n## RPY Comparison Charts\n\n")
            for cf in chart_files:
                handle.write("![{}](images/{})\n\n".format(cf, cf))

        if rpy_stats_content:
            handle.write("\n" + rpy_stats_content + "\n")

        handle.write("\n## Failed trips\n\n")
        if not failures:
            handle.write("(none)\n")
        else:
            for r in failures:
                handle.write(
                    "- {0}: {1}\n".format(r.get("trip", "?"), r.get("error", "unknown"))
                )

    print("[info] Wrote {}".format(report_md))

    feishu_url = getattr(args, "feishu_webhook", "")
    if feishu_url:
        try:
            from tools.feishu_trip_sync import format_report_for_feishu, push_to_feishu
            title, lines = format_report_for_feishu(report_md)
            push_to_feishu(feishu_url, title, lines)
        except Exception as exc:
            print("[warn] Feishu push failed: {}".format(exc))

    cleanup_cache = getattr(args, "cleanup_cache", False)
    if cleanup_cache and input_format == "remote" and hasattr(args, "trips_base"):
        print("[info] Cleaning up cached trip data (cleanup_cache=true)...")
        for tp in trips_payload:
            cache_dir = os.path.join(args.trips_base, tp["trip_name"])
            if os.path.isdir(cache_dir):
                try:
                    shutil.rmtree(cache_dir)
                    print("[info] Removed cache: {}".format(cache_dir))
                except Exception as exc:
                    print("[warn] Failed to remove cache {}: {}".format(cache_dir, exc))

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()