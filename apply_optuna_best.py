#!/usr/bin/env python3
"""Read Optuna study best trial and generate v49 YAML config snippet."""

import argparse
import json
from pathlib import Path
from typing import Optional

import optuna
import yaml

BEVCALIB_ROOT = Path(__file__).parent.resolve()
LOG_BASE = BEVCALIB_ROOT / "logs" / "optuna_search"

PARAM_MAP = {
    "gin_channels": ("gin_channels", int),
    "cf_n_groups": ("cf_n_groups", int),
    "pitch_vertical_bands": ("pitch_vertical_bands", int),
    "lr": ("learning_rate", float),
    "head_dropout": ("head_dropout", float),
    "consistency_loss_weight": ("consistency_loss_weight", float),
    "consistency_loss_start_epoch": ("consistency_loss_start_epoch", int),
    "overcorrection_penalty": ("overcorrection_penalty", float),
    "multi_scale_perturb": ("multi_scale_perturb", str),
    "axis_weights": ("axis_weights", str),
    "cosine_T0": ("cosine_T0", int),
    "warmup_epochs": ("warmup_epochs", int),
    "tinit_dropout_prob": ("tinit_dropout_prob", float),
    "zero_perturbation_prob": ("zero_perturbation_prob", float),
    "gin_gate_reg_weight": ("gin_gate_reg_weight", float),
    "pitch_aux_weight": ("pitch_aux_weight", float),
    "conditional_consistency_threshold": ("conditional_consistency_threshold", float),
    "quat_bias_reset_alpha": ("quat_bias_reset_alpha", float),
    "decoder_pool_mode": ("decoder_pool_mode", str),
}


def load_study(study_name: str, db_path: str = None) -> optuna.Study:
    if db_path is None:
        db_path = LOG_BASE / study_name / "optuna_study.db"
    storage = f"sqlite:///{db_path}"
    return optuna.load_study(study_name=study_name, storage=storage)


def _load_trial_result_params(study_name: str, trial_number: int) -> Optional[dict]:
    """Load exact training params saved by optuna_search objective."""
    result_path = LOG_BASE / study_name / f"trial_{trial_number:04d}" / "result.json"
    if not result_path.is_file():
        return None
    with open(result_path) as f:
        saved = json.load(f).get("params", {})
    return {k: v for k, v in saved.items() if not k.startswith("_")}


def _derive_composite_params(raw: dict) -> dict:
    """Rebuild composite params from raw Optuna search-space keys."""
    out = {}
    if "pitch_weight" in raw:
        out["axis_weights"] = f"1.5,{raw['pitch_weight']:.1f},1.5"
    if all(k in raw for k in ("ms_small_ratio", "ms_mid_ratio", "ms_large_ratio")):
        out["multi_scale_perturb"] = (
            f"0.5:{raw['ms_small_ratio']:.2f},"
            f"1.0:{raw['ms_mid_ratio']:.2f},"
            f"2.0:{raw['ms_large_ratio']:.2f}"
        )
    return out


def trial_to_params(trial: optuna.trial.FrozenTrial, study_name: str = None) -> dict:
    raw = _load_trial_result_params(study_name, trial.number) if study_name else None
    if raw is None:
        raw = dict(trial.params)

    out = {}
    for opt_key, (yaml_key, cast) in PARAM_MAP.items():
        if opt_key in raw:
            out[yaml_key] = cast(raw[opt_key])

    out.update(_derive_composite_params(raw))

    gin = raw.get("gin_channels", trial.params.get("gin_channels", 0))
    if gin == 0:
        out["use_gated_instance_norm"] = 0
        out["gin_gate_reg_weight"] = 0.0
    else:
        out["use_gated_instance_norm"] = 1
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study_name", default="bevcalib_parallel_v49")
    parser.add_argument("--db", default=None)
    parser.add_argument("--output", default="configs/v49_optuna_best.yaml")
    parser.add_argument("--top_k", type=int, default=1,
                        help="Include top-K trials as separate experiments")
    args = parser.parse_args()

    study = load_study(args.study_name, args.db)
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
    if not completed:
        print("No completed trials yet.")
        return

    sorted_trials = sorted(completed, key=lambda t: t.value)[:args.top_k]

    base_cfg_path = BEVCALIB_ROOT / "configs" / "v49_cf_bev_r_quick.yaml"
    with open(base_cfg_path) as f:
        base = yaml.safe_load(f)

    experiments = []
    for i, trial in enumerate(sorted_trials):
        exp = {
            "name": f"v49_optuna_best_{i+1}_S1_quick",
            "description": (
                f"Optuna best trial #{trial.number}: "
                f"MEDW max(R,P,Y)={trial.value:.4f}°"
            ),
            "dataset": "all",
            "version": f"v49_optuna_best_{i+1}_S1_quick",
            "params": dict(base["defaults"]["params"]),
        }
        exp["params"].update(trial_to_params(trial, args.study_name))
        experiments.append(exp)

    out_cfg = {
        "global": base["global"],
        "defaults": base["defaults"],
        "experiments": experiments,
    }

    out_path = BEVCALIB_ROOT / args.output
    with open(out_path, "w") as f:
        yaml.dump(out_cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"Wrote {len(experiments)} experiment(s) to {out_path}")
    best = sorted_trials[0]
    print(f"\nBest trial #{best.number}: MEDW max(R,P,Y)={best.value:.4f}°")
    print(f"  Roll={best.user_attrs.get('best_roll')}, "
          f"Pitch={best.user_attrs.get('best_pitch')}, "
          f"Yaw={best.user_attrs.get('best_yaw')}")
    print(f"  Key params: gin={best.params.get('gin_channels')}, "
          f"lr={best.params.get('lr'):.2e}, "
          f"pool={best.params.get('decoder_pool_mode')}")


if __name__ == "__main__":
    main()
