#!/usr/bin/env python3
"""
并行评估三个模型在测试数据上的泛化能力
"""
import subprocess
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

BASE_DIR = "/mnt/drtraining/user/dahailu/code/BEVCalib"
TEST_DATA = "/mnt/drtraining/user/dahailu/data/bevcalib/test_data"

MODELS = [
    {
        "name": "z=1",
        "zbound_step": "20.0",
        "model_dir": "model_small_5deg_v4-z1",
    },
    {
        "name": "z=5",
        "zbound_step": "4.0",
        "model_dir": "model_small_5deg_v4-z5",
    },
    {
        "name": "z=10",
        "zbound_step": "2.0",
        "model_dir": "model_small_5deg_v4-z10",
    },
]


def evaluate_model(model_config):
    """评估单个模型"""
    name = model_config["name"]
    zbound_step = model_config["zbound_step"]
    model_dir = model_config["model_dir"]
    
    ckpt_path = f"{BASE_DIR}/logs/B26A/{model_dir}/B26A_scratch/checkpoint/ckpt_400.pth"
    output_dir = f"{BASE_DIR}/logs/B26A/{model_dir}/test_data_eval"
    
    if not os.path.exists(ckpt_path):
        return {"name": name, "status": "failed", "error": f"Checkpoint not found: {ckpt_path}"}
    
    print(f"[{name}] Starting evaluation...")
    print(f"[{name}]   BEV_ZBOUND_STEP: {zbound_step}")
    print(f"[{name}]   Output: {output_dir}")
    
    env = os.environ.copy()
    env["BEV_ZBOUND_STEP"] = zbound_step
    
    cmd = [
        "conda", "run", "-n", "bevcalib", "python",
        f"{BASE_DIR}/evaluate_checkpoint.py",
        "--ckpt_path", ckpt_path,
        "--dataset_root", TEST_DATA,
        "--output_dir", output_dir,
        "--angle_range_deg", "5",
        "--trans_range", "0.3",
        "--use_full_dataset",
        "--max_batches", "0",
        "--vis_interval", "10",
        "--batch_size", "4",
        "--rotation_only", "1",
    ]
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            env=env,
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            # 提取结果摘要
            errors_file = f"{output_dir}/extrinsics_and_errors.txt"
            if os.path.exists(errors_file):
                with open(errors_file, 'r') as f:
                    lines = f.readlines()
                    summary = ''.join(lines[-20:])
            else:
                summary = "Results file not found"
            
            print(f"[{name}] ✓ Evaluation complete ({elapsed:.1f}s)")
            return {
                "name": name,
                "status": "success",
                "elapsed": elapsed,
                "summary": summary,
            }
        else:
            print(f"[{name}] ✗ Evaluation failed ({elapsed:.1f}s)")
            print(f"[{name}] Error: {result.stderr[:500]}")
            return {
                "name": name,
                "status": "failed",
                "elapsed": elapsed,
                "error": result.stderr,
            }
    
    except subprocess.TimeoutExpired:
        print(f"[{name}] ✗ Evaluation timed out")
        return {"name": name, "status": "timeout"}
    except Exception as e:
        print(f"[{name}] ✗ Evaluation error: {e}")
        return {"name": name, "status": "error", "error": str(e)}


def main():
    """主函数"""
    print("=" * 80)
    print("Parallel Evaluation of BEVCalib Models on Test Data")
    print("=" * 80)
    print()
    
    # 使用进程池并行评估
    with ProcessPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(evaluate_model, model): model for model in MODELS}
        
        results = []
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
    
    print()
    print("=" * 80)
    print("Evaluation Results Summary")
    print("=" * 80)
    print()
    
    for result in sorted(results, key=lambda x: x["name"]):
        print(f"Model: {result['name']}")
        print(f"  Status: {result['status']}")
        if result['status'] == 'success':
            print(f"  Time: {result['elapsed']:.1f}s")
            print("  Summary:")
            for line in result['summary'].split('\n')[:15]:
                if line.strip():
                    print(f"    {line}")
        elif 'error' in result:
            print(f"  Error: {result['error'][:200]}")
        print()


if __name__ == "__main__":
    main()
