#!/usr/bin/env python3
"""
BEVCalib Optuna + ASHA 超参数搜索

目标: 找到使 MEDW200 RPY 全部 < 0.1 度的最优网络架构和训练参数
策略: Bayesian Optimization (TPE) + Successive Halving (ASHA) 早停

用法:
  # 并行 DDP 模式(推荐): 8卡拆成2组4卡, 2 trials 同时训练
  conda run -n bevcalib310 python3 optuna_search.py --n_parallel 2 --n_trials 40 --max_epochs 40

  # 独占模式: 8 卡 DDP, 30 trials 串行
  conda run -n bevcalib310 python3 optuna_search.py --n_trials 30 --max_epochs 80

  # 从已有 study 继续
  conda run -n bevcalib310 python3 optuna_search.py --study_name bevcalib_v49 --n_trials 20

  # 只查看已有结果
  conda run -n bevcalib310 python3 optuna_search.py --report_only
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import optuna
from optuna.pruners import SuccessiveHalvingPruner
from optuna.samplers import TPESampler

BEVCALIB_ROOT = Path(__file__).parent.resolve()
LOG_BASE = BEVCALIB_ROOT / "logs" / "optuna_search"
PRETRAIN_CKPT = "logs/all_training_data/model_small_5deg_v45c_cf_bev_r_S1_quick/all_training_data_scratch/checkpoint/ckpt_best_val.pth"

# GPU group pool for parallel DDP mode
_gpu_group_lock = threading.Lock()
_gpu_groups: list = []          # each element: list[int] of GPU IDs
_gpu_group_sem: threading.Semaphore = threading.Semaphore(0)


def _init_gpu_groups(total_gpus: int, n_groups: int):
    """Split total_gpus into n_groups equal-sized DDP groups."""
    global _gpu_groups, _gpu_group_sem
    per_group = total_gpus // n_groups
    _gpu_groups = [
        list(range(i * per_group, (i + 1) * per_group))
        for i in range(n_groups)
    ]
    _gpu_group_sem = threading.Semaphore(n_groups)


def _acquire_gpu_group() -> list:
    """Block until a GPU group is available, return list of GPU IDs."""
    _gpu_group_sem.acquire()
    with _gpu_group_lock:
        return _gpu_groups.pop(0)


def _release_gpu_group(group: list):
    with _gpu_group_lock:
        _gpu_groups.append(group)
    _gpu_group_sem.release()


def define_search_space(trial: optuna.Trial) -> dict:
    """Define the hyperparameter search space."""
    params = {}

    # --- Architecture ---
    params["gin_channels"] = trial.suggest_categorical("gin_channels", [0, 32, 64, 128])
    params["use_gated_instance_norm"] = 1 if params["gin_channels"] > 0 else 0
    params["cf_n_groups"] = trial.suggest_categorical("cf_n_groups", [128, 256])
    params["pitch_vertical_bands"] = trial.suggest_int("pitch_vertical_bands", 2, 5)

    # --- Training strategy ---
    params["lr"] = trial.suggest_float("lr", 3e-5, 3e-4, log=True)
    params["head_dropout"] = trial.suggest_float("head_dropout", 0.05, 0.25, step=0.05)
    params["consistency_loss_weight"] = trial.suggest_float(
        "consistency_loss_weight", 0.0, 0.8, step=0.1
    )
    params["consistency_loss_start_epoch"] = trial.suggest_categorical(
        "consistency_loss_start_epoch", [5, 10, 15]
    )
    params["overcorrection_penalty"] = trial.suggest_float(
        "overcorrection_penalty", 1.0, 3.0, step=0.5
    )

    # --- Multi-Scale perturbation ---
    ms_small = trial.suggest_float("ms_small_ratio", 0.05, 0.20, step=0.05)
    ms_mid = trial.suggest_float("ms_mid_ratio", 0.05, 0.15, step=0.05)
    ms_large = trial.suggest_float("ms_large_ratio", 0.05, 0.15, step=0.05)
    params["multi_scale_perturb"] = f"0.5:{ms_small:.2f},1.0:{ms_mid:.2f},2.0:{ms_large:.2f}"

    # --- Axis weights ---
    pitch_w = trial.suggest_float("pitch_weight", 2.0, 6.0, step=0.5)
    params["axis_weights"] = f"1.5,{pitch_w:.1f},1.5"

    # --- Cosine schedule ---
    params["cosine_T0"] = trial.suggest_categorical("cosine_T0", [30, 40, 50])
    params["warmup_epochs"] = trial.suggest_int("warmup_epochs", 5, 12)

    # --- Augmentation ---
    params["tinit_dropout_prob"] = trial.suggest_float(
        "tinit_dropout_prob", 0.1, 0.5, step=0.1
    )
    params["zero_perturbation_prob"] = trial.suggest_float(
        "zero_perturbation_prob", 0.15, 0.35, step=0.05
    )

    # --- GIN gate reg (only if GIN enabled) ---
    if params["gin_channels"] > 0:
        params["gin_gate_reg_weight"] = trial.suggest_float(
            "gin_gate_reg_weight", 0.0, 0.2, step=0.02
        )
    else:
        params["gin_gate_reg_weight"] = 0.0

    # --- Pitch weight scaling for balanced axis training ---
    params["pitch_aux_weight"] = trial.suggest_float(
        "pitch_aux_weight", 0.2, 0.6, step=0.1
    )

    # --- V49 improvements ---
    params["conditional_consistency_threshold"] = trial.suggest_float(
        "conditional_consistency_threshold", 0.0, 3.0, step=0.5
    )
    params["quat_bias_reset_alpha"] = trial.suggest_float(
        "quat_bias_reset_alpha", 0.0, 0.5, step=0.1
    )
    params["decoder_pool_mode"] = trial.suggest_categorical(
        "decoder_pool_mode", ["mean", "attention"]
    )

    return params


def build_train_command(params: dict, trial_dir: str, max_epochs: int,
                        n_gpus: int, batch_size: int = 16,
                        master_port: int = 0) -> list:
    """Build the training command from search parameters."""
    version = f"optuna_trial_{os.path.basename(trial_dir)}"

    cmd = [
        "bash", "start_training.sh", "all", version,
        "--angle", "5",
        "--trans", "0.0",
        "--bs", str(batch_size),
        "--lr", str(params["lr"]),
        "--ddp", str(n_gpus),
        "--rotation_only",
        "--enable_axis_loss",
        "--weight_axis_rotation", "0.5",
        "--lr_schedule", "cosine_warm_restarts",
        "--warmup_epochs", str(params["warmup_epochs"]),
        "--backbone_lr_scale", "0.1",
        "--cosine_T0", str(params["cosine_T0"]),
        "--cosine_Tmult", "2",
        "--head_dropout", str(params["head_dropout"]),
        "--perturb_distribution", "magnitude_balanced",
        "--per_axis_prob", "0.4",
        "--axis_weights", params["axis_weights"],
        "--augment_color_jitter", "0.3",
        "--augment_intrinsic", "0.03",
        "--augment_intrinsic_cxcy", "0.015",
        "--eval_angle", "5",
        "--early_stopping_patience", "30",
        "--seed", "42",
        "--pretrain_ckpt", PRETRAIN_CKPT,
        "--no_amp", "1",
        "--num_epochs", str(max_epochs),
        "--save_ckpt_per_epoches", "10",
        "--max_frames_per_seq", str(params.get("_max_frames", 500)),
        "--eval_epoches", "10",
        "--enable_vis", "0",
        "--enable_ckpt_eval", "0",
        "--enable_medw_eval", "1",
        "--enable_jacobian_eval", "1",
        "--jacobian_eval_angle_deg", "3.0",
        "--jacobian_eval_batches", "2",
        "--jacobian_eval_n_probes", "3",
        "--medw_eval_max_frames", "200",
        "--enable_dual_gate_ckpt", "1",
        "--dual_gate_jacobian_min", "0.85",
        "--dual_gate_medw_max", "0.1",
        "--use_pitch_branch", "1",
        "--pitch_aux_weight", "0.4",
        "--use_dla", "1",
        "--use_pitch_fusion", "1",
        "--use_instance_norm", "0",
        "--augment_mount_jitter_prob", "0.4",
        "--augment_mount_jitter_rot_sigma", "1.5",
        "--augment_mount_jitter_trans_sigma", "0.015",
        "--backbone_type", "swin",
        "--tinit_dropout_prob", str(params["tinit_dropout_prob"]),
        "--consistency_loss_weight", str(params["consistency_loss_weight"]),
        "--consistency_loss_start_epoch", str(params["consistency_loss_start_epoch"]),
        "--continuous_tinit_noise", "1",
        "--continuous_noise_max_deg", "15.0",
        "--fusion_backend", "cf_bev_r",
        "--cf_feat_dim", "256",
        "--cf_n_groups", str(params["cf_n_groups"]),
        "--cf_knn", "8",
        "--cf_corr_heads", "4",
        "--cf_corr_radius", "4",
        "--cf_num_queries", "6",
        "--cf_encoder_layers", "2",
        "--cf_decoder_layers", "4",
        "--use_rocr", "1",
        "--rocr_dropout", "0.3",
        "--rocr_center_bias", "0.5",
        "--rocr_detach_epochs", "999",
        "--quat_norm_weight", "0.5",
        "--corr_alignment_weight", "0.2",
        "--corr_alignment_warmup", "20",
        "--seq_consistency_weight", "0.1",
        "--seq_consistency_start_epoch", "999",
        "--corr_window_mode", "adaptive",
        "--max_pcd_points", "16384",
        "--zero_perturbation_prob", str(params["zero_perturbation_prob"]),
        "--symmetric_perturb", "1",
        "--overcorrection_penalty", str(params["overcorrection_penalty"]),
        "--use_magnitude_head", "1",
        "--magnitude_loss_weight", "0.3",
        "--pitch_aux_weight", str(params.get("pitch_aux_weight", 0.4)),
        "--conditional_consistency_threshold", str(params.get("conditional_consistency_threshold", 0.0)),
        "--quat_bias_reset_alpha", str(params.get("quat_bias_reset_alpha", 0.0)),
        "--decoder_pool_mode", params.get("decoder_pool_mode", "mean"),
        "--use_gated_instance_norm", str(params["use_gated_instance_norm"]),
        "--gin_channels", str(params["gin_channels"]),
        "--pitch_vertical_bands", str(params["pitch_vertical_bands"]),
        "--gin_gate_reg_target", "0.5",
        "--gin_gate_reg_weight", str(params["gin_gate_reg_weight"]),
        "--multi_scale_perturb", params["multi_scale_perturb"],
        "--pose_aware_sampling",
        "--fg", "--no-tb",
        "--nnodes", "1",
    ]
    if master_port > 0:
        cmd.extend(["--master_port", str(master_port)])
    return cmd


def parse_train_log(log_path: str) -> list:
    """Parse training log to extract per-epoch MEDW200 and Jacobian results.

    Returns list of dicts sorted by epoch.
    """
    results = []
    if not os.path.isfile(log_path):
        return results

    medw_re = re.compile(
        r"Epoch \[(\d+)/\d+\], MEDW200 \(val reuse\): "
        r"max\(R,P,Y\)=([0-9.]+)° \(R:([0-9.]+) P:([0-9.]+) Y:([0-9.]+)"
    )
    jac_re = re.compile(
        r"Epoch \[(\d+)/\d+\], Jacobian.*Overall=([0-9.]+)"
    )
    val_re = re.compile(
        r"Epoch \[(\d+)/\d+\], Train Pose Error - Rot: ([0-9.]+)°"
    )

    epoch_data = {}

    with open(log_path, "r") as f:
        for line in f:
            m = medw_re.search(line)
            if m:
                ep = int(m.group(1))
                if ep not in epoch_data:
                    epoch_data[ep] = {"epoch": ep}
                epoch_data[ep].update({
                    "medw_max_rpy": float(m.group(2)),
                    "medw_roll": float(m.group(3)),
                    "medw_pitch": float(m.group(4)),
                    "medw_yaw": float(m.group(5)),
                })

            m = jac_re.search(line)
            if m:
                ep = int(m.group(1))
                if ep not in epoch_data:
                    epoch_data[ep] = {"epoch": ep}
                epoch_data[ep]["jacobian"] = float(m.group(2))

            m = val_re.search(line)
            if m:
                ep = int(m.group(1))
                if ep not in epoch_data:
                    epoch_data[ep] = {"epoch": ep}
                epoch_data[ep]["train_rot"] = float(m.group(2))

    results = sorted(epoch_data.values(), key=lambda x: x["epoch"])
    return results


def objective(trial: optuna.Trial, args) -> float:
    """Optuna objective function: train BEVCalib and return best MEDW200 max(R,P,Y)."""
    params = define_search_space(trial)

    trial_name = f"trial_{trial.number:04d}"
    trial_dir = str(LOG_BASE / args.study_name / trial_name)
    os.makedirs(trial_dir, exist_ok=True)

    with open(os.path.join(trial_dir, "params.json"), "w") as f:
        json.dump(params, f, indent=2)

    log_dir = str(
        BEVCALIB_ROOT
        / "logs"
        / "all_training_data"
        / f"model_small_5deg_optuna_trial_{trial_name}"
    )
    train_log = os.path.join(log_dir, "train.log")
    ckpt_dir = os.path.join(log_dir, "all_training_data_scratch", "checkpoint")

    gpu_group = None
    if args.n_parallel > 1:
        gpu_group = _acquire_gpu_group()
        n_gpus = len(gpu_group)
        batch_size = args.batch_size
        params["_max_frames"] = args.max_frames_per_seq
    else:
        n_gpus = args.n_gpus
        batch_size = args.batch_size
        params["_max_frames"] = args.max_frames_per_seq

    group_tag = ",".join(map(str, gpu_group)) if gpu_group else "all"
    trial_port = 40000 + trial.number * 100 + (gpu_group[0] if gpu_group else 0)
    cmd = build_train_command(params, trial_name, args.max_epochs, n_gpus,
                              batch_size=batch_size, master_port=trial_port)

    env = os.environ.copy()
    env["BATCH_MODE"] = "1"
    env["BEV_ZBOUND_STEP"] = "4.0"
    env["USE_DRCV_BACKEND"] = "0"
    env["HF_HUB_OFFLINE"] = "1"
    env["NCCL_IB_TIMEOUT"] = "50"
    env["NCCL_SOCKET_IFNAME"] = "eth0"
    env["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"

    if gpu_group is not None:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_group))

    gpu_info = (f"DDP={n_gpus} GPUs=[{group_tag}], bs={batch_size}, "
                f"frames={params['_max_frames']}")
    print(f"\n{'='*60}")
    print(f"Trial {trial.number}: Starting training ({gpu_info})")
    print(f"  GIN={params['gin_channels']}, cf_groups={params['cf_n_groups']}, "
          f"lr={params['lr']:.2e}, pitch_bands={params['pitch_vertical_bands']}")
    print(f"  Log: {train_log}")
    print(f"{'='*60}\n")

    trial_stdout_path = os.path.join(trial_dir, "stdout.log")
    trial_stdout = open(trial_stdout_path, "w")
    proc = subprocess.Popen(
        cmd,
        cwd=str(BEVCALIB_ROOT),
        env=env,
        stdout=trial_stdout,
        stderr=subprocess.STDOUT,
    )

    best_medw = float("inf")
    last_checked_epoch = 0

    try:
        while proc.poll() is None:
            time.sleep(30)

            results = parse_train_log(train_log)
            for r in results:
                ep = r["epoch"]
                if ep <= last_checked_epoch:
                    continue
                last_checked_epoch = ep

                medw = r.get("medw_max_rpy", float("inf"))
                jac = r.get("jacobian", 0.0)

                if medw < best_medw:
                    best_medw = medw

                score = medw - 0.1 * jac
                trial.report(score, ep)

                print(
                    f"  Trial {trial.number} ep{ep} [{group_tag}]: "
                    f"MEDW={medw:.4f}° Jac={jac:.3f} best={best_medw:.4f}°"
                )

                if trial.should_prune():
                    print(f"  Trial {trial.number}: PRUNED at epoch {ep}")
                    proc.terminate()
                    proc.wait(timeout=60)
                    raise optuna.TrialPruned()

    except optuna.TrialPruned:
        trial_stdout.close()
        if gpu_group is not None:
            _release_gpu_group(gpu_group)
        raise
    except Exception as e:
        print(f"  Trial {trial.number}: ERROR - {e}")
        proc.terminate()
        proc.wait(timeout=60)
        trial_stdout.close()
        if gpu_group is not None:
            _release_gpu_group(gpu_group)
        return float("inf")

    proc.wait()
    trial_stdout.close()

    if gpu_group is not None:
        _release_gpu_group(gpu_group)

    medw_summary = os.path.join(ckpt_dir, "medw_eval_summary.json")
    if os.path.isfile(medw_summary):
        with open(medw_summary) as f:
            summary = json.load(f)
        best_medw = summary.get("best_medw_max_rpy", best_medw)
        best_roll = summary.get("best_medw_roll", 0)
        best_pitch = summary.get("best_medw_pitch", 0)
        best_yaw = summary.get("best_medw_yaw", 0)
        best_ep = summary.get("best_epoch", -1)

        trial.set_user_attr("best_roll", best_roll)
        trial.set_user_attr("best_pitch", best_pitch)
        trial.set_user_attr("best_yaw", best_yaw)
        trial.set_user_attr("best_epoch", best_ep)

        print(
            f"\n  Trial {trial.number} COMPLETE: "
            f"MEDW max(R,P,Y)={best_medw:.4f}° "
            f"(R={best_roll:.4f} P={best_pitch:.4f} Y={best_yaw:.4f}) "
            f"@ epoch {best_ep}"
        )

    result_path = os.path.join(trial_dir, "result.json")
    with open(result_path, "w") as f:
        json.dump({
            "trial_number": trial.number,
            "params": params,
            "best_medw_max_rpy": best_medw,
            "best_roll": trial.user_attrs.get("best_roll"),
            "best_pitch": trial.user_attrs.get("best_pitch"),
            "best_yaw": trial.user_attrs.get("best_yaw"),
            "best_epoch": trial.user_attrs.get("best_epoch"),
        }, f, indent=2)

    return best_medw


def print_report(study: optuna.Study):
    """Print a summary report of the study."""
    print(f"\n{'='*80}")
    print(f"Optuna Study Report: {study.study_name}")
    print(f"{'='*80}")

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    failed = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]

    print(f"\nTrials: {len(study.trials)} total, "
          f"{len(completed)} complete, {len(pruned)} pruned, {len(failed)} failed")

    if not completed:
        print("No completed trials yet.")
        return

    print(f"\nBest trial #{study.best_trial.number}:")
    print(f"  MEDW max(R,P,Y) = {study.best_value:.4f}°")
    roll = study.best_trial.user_attrs.get("best_roll", "N/A")
    pitch = study.best_trial.user_attrs.get("best_pitch", "N/A")
    yaw = study.best_trial.user_attrs.get("best_yaw", "N/A")
    print(f"  Roll={roll}, Pitch={pitch}, Yaw={yaw}")
    print(f"  Best params:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")

    print(f"\nTop-5 trials:")
    sorted_trials = sorted(completed, key=lambda t: t.value)
    for i, t in enumerate(sorted_trials[:5]):
        r = t.user_attrs.get("best_roll", "?")
        p = t.user_attrs.get("best_pitch", "?")
        y = t.user_attrs.get("best_yaw", "?")
        gin = t.params.get("gin_channels", "?")
        lr = t.params.get("lr", "?")
        print(f"  {i+1}. Trial#{t.number}: {t.value:.4f}° "
              f"(R={r} P={p} Y={y}) "
              f"GIN={gin} lr={lr}")

    target_met = [
        t for t in completed
        if t.user_attrs.get("best_roll", 1) < 0.1
        and t.user_attrs.get("best_pitch", 1) < 0.1
        and t.user_attrs.get("best_yaw", 1) < 0.1
    ]
    if target_met:
        print(f"\nTrials meeting RPY < 0.1° target: {len(target_met)}")
        for t in target_met:
            print(f"  Trial#{t.number}: max(R,P,Y)={t.value:.4f}°")
    else:
        print(f"\nNo trials yet meet the RPY < 0.1° target.")

    importances = optuna.importance.get_param_importances(study)
    print(f"\nParameter importance:")
    for param, imp in importances.items():
        print(f"  {param}: {imp:.3f}")


def main():
    parser = argparse.ArgumentParser(description="BEVCalib Optuna HPO Search")
    parser.add_argument("--study_name", default="bevcalib_v49",
                        help="Optuna study name (for persistence)")
    parser.add_argument("--n_trials", type=int, default=30,
                        help="Number of trials to run")
    parser.add_argument("--max_epochs", type=int, default=80,
                        help="Max epochs per trial (ASHA may stop earlier)")
    parser.add_argument("--n_gpus", type=int, default=8,
                        help="Total GPUs available (default: 8)")
    parser.add_argument("--n_parallel", type=int, default=2,
                        help="Parallel trials; GPUs split into n_parallel DDP groups "
                             "(default: 2 → 2×4-GPU DDP)")
    parser.add_argument("--batch_size", type=int, default=20,
                        help="Per-GPU batch size (default: 20)")
    parser.add_argument("--max_frames_per_seq", type=int, default=200,
                        help="Max frames per sequence (default: 200 for fast HPO)")
    parser.add_argument("--report_only", action="store_true",
                        help="Only print report from existing study")
    parser.add_argument("--db", default=None,
                        help="SQLite DB path for study persistence")
    args = parser.parse_args()

    os.makedirs(LOG_BASE / args.study_name, exist_ok=True)

    storage = args.db
    if storage is None:
        db_path = LOG_BASE / args.study_name / "optuna_study.db"
        storage = f"sqlite:///{db_path}"

    pruner = SuccessiveHalvingPruner(
        min_resource=10,
        reduction_factor=3,
        min_early_stopping_rate=0,
    )
    sampler = TPESampler(seed=42, n_startup_trials=5)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
    )

    if args.report_only:
        print_report(study)
        return

    if args.n_parallel > 1:
        if args.n_gpus % args.n_parallel != 0:
            raise ValueError(
                f"n_gpus ({args.n_gpus}) must be divisible by n_parallel ({args.n_parallel})"
            )
        _init_gpu_groups(args.n_gpus, args.n_parallel)
        per_group = args.n_gpus // args.n_parallel
        n_jobs = args.n_parallel
        mode_desc = (f"parallel DDP: {args.n_parallel}×{per_group}-GPU, "
                     f"bs={args.batch_size}, frames={args.max_frames_per_seq}")
    else:
        n_jobs = 1
        mode_desc = (f"serial DDP: {args.n_gpus}-GPU, bs={args.batch_size}, "
                     f"frames={args.max_frames_per_seq}")

    print(f"Starting Optuna search: {args.n_trials} trials, max {args.max_epochs} epochs each")
    print(f"Mode: {mode_desc}")
    print(f"Study: {args.study_name}, DB: {storage}")
    print(f"Target: MEDW200 RPY all < 0.1°")

    study.optimize(
        lambda trial: objective(trial, args),
        n_trials=args.n_trials,
        n_jobs=n_jobs,
        catch=(Exception,),
    )

    print_report(study)


if __name__ == "__main__":
    main()
