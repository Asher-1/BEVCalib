#!/usr/bin/env python3
"""
BEVCalib 实验分析统一入口脚本

用法:
    python analyze_experiments.py [--config CONFIG_FILE] [--skip-test] [--only-train]

示例:
    # 使用默认配置分析实验
    python analyze_experiments.py
    
    # 使用自定义配置文件
    python analyze_experiments.py --config my_config.yaml
    
    # 只分析训练数据，跳过测试集评估
    python analyze_experiments.py --only-train
    
    # 跳过测试集评估（但会尝试读取已有的评估结果）
    python analyze_experiments.py --skip-test
"""
import os
import sys
import argparse
import yaml
from pathlib import Path

# 添加utils路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))

from analysis import TrainingAnalyzer, TestEvaluator, ReportGenerator, Visualizer


def load_config(config_file: str) -> dict:
    """加载配置文件"""
    if not os.path.exists(config_file):
        print(f"✗ Config file not found: {config_file}")
        sys.exit(1)
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def analyze_training(config: dict) -> dict:
    """分析训练数据"""
    print("\n" + "="*80)
    print("一、训练日志分析")
    print("="*80 + "\n")
    
    logs_base = config['paths']['logs_base']
    experiments_config = config['experiments']
    
    training_results = {}
    
    for exp_config in experiments_config:
        name = exp_config['name']
        model_dir = exp_config['model_dir']
        log_path = os.path.join(logs_base, model_dir, "train.log")
        
        print(f"分析实验: {name}")
        print(f"  日志: {log_path}")
        
        if not os.path.exists(log_path):
            print(f"  ⚠ 日志文件不存在，跳过")
            continue
        
        try:
            analyzer = TrainingAnalyzer(log_path)
            data = analyzer.parse_log()
            
            training_results[name] = {
                'data': data,
                'final_metrics': analyzer.get_final_metrics(),
                'convergence': analyzer.get_convergence_data(
                    config['report'].get('include_convergence_milestones')
                )
            }
            
            if data['train']['epoch']:
                print(f"  ✓ 提取了 {len(data['train']['epoch'])} 个训练epochs")
                final = analyzer.get_final_metrics()
                if final:
                    print(f"    最终: Trans={final.get('trans_error', 0):.4f}m, "
                          f"Lat={final.get('lat', 0):.4f}m, "
                          f"Rot={final.get('rot_error', 0):.2f}°")
            else:
                print(f"  ⚠ 未找到训练数据")
        
        except Exception as e:
            print(f"  ✗ 分析失败: {e}")
    
    return training_results


def evaluate_test_data(config: dict, training_results: dict) -> dict:
    """评估测试数据"""
    print("\n" + "="*80)
    print("二、测试集评估")
    print("="*80 + "\n")
    
    bevcalib_root = config['paths']['bevcalib_root']
    test_data_root = config['paths']['test_data']
    logs_base = config['paths']['logs_base']
    
    evaluator = TestEvaluator(bevcalib_root, test_data_root)
    test_results = {}
    
    eval_config = config['evaluation']
    experiments_config = config['experiments']
    
    for exp_config in experiments_config:
        name = exp_config['name']
        
        # 跳过没有训练数据的实验
        if name not in training_results:
            continue
        
        model_dir = exp_config['model_dir']
        checkpoint = exp_config.get('checkpoint', 'ckpt_400.pth')
        zbound_step = exp_config['zbound_step']
        
        ckpt_path = os.path.join(logs_base, model_dir, "B26A_scratch/checkpoint", checkpoint)
        output_dir = os.path.join(logs_base, model_dir, "test_data_eval")
        
        print(f"评估实验: {name}")
        print(f"  Checkpoint: {checkpoint}")
        print(f"  BEV_ZBOUND_STEP: {zbound_step}")
        
        # 检查是否已有评估结果
        if os.path.exists(os.path.join(output_dir, "extrinsics_and_errors.txt")):
            print(f"  ℹ 发现已有评估结果，直接读取...")
            test_results[name] = evaluator.parse_evaluation_results(output_dir)
            if test_results[name]:
                print(f"  ✓ Total Rot: {test_results[name]['total']['mean']:.3f}° "
                      f"± {test_results[name]['total']['std']:.3f}°")
            else:
                print(f"  ⚠ 解析评估结果失败")
            continue
        
        # 运行评估
        if not os.path.exists(ckpt_path):
            print(f"  ⚠ Checkpoint不存在: {checkpoint}")
            test_results[name] = None
            continue
        
        success = evaluator.evaluate_checkpoint(
            ckpt_path=ckpt_path,
            output_dir=output_dir,
            zbound_step=zbound_step,
            angle_range=eval_config['angle_range_deg'],
            trans_range=eval_config['trans_range'],
            batch_size=eval_config['batch_size'],
            rotation_only=eval_config['rotation_only'],
            vis_interval=eval_config['vis_interval'],
            use_conda=eval_config['use_conda'],
            conda_env=eval_config['conda_env']
        )
        
        if success:
            test_results[name] = evaluator.parse_evaluation_results(output_dir)
            if test_results[name]:
                print(f"  ✓ Total Rot: {test_results[name]['total']['mean']:.3f}° "
                      f"± {test_results[name]['total']['std']:.3f}°")
        else:
            test_results[name] = None
    
    return test_results


def generate_visualizations(
    config: dict,
    training_results: dict,
    test_results: dict
):
    """生成可视化图表"""
    if not config['visualization']['generate_plots']:
        print("\n跳过可视化生成（配置中disabled）")
        return
    
    print("\n" + "="*80)
    print("三、生成可视化图表")
    print("="*80 + "\n")
    
    output_dir = config['paths']['output_dir']
    viz_config = config['visualization']
    
    visualizer = Visualizer(output_dir, style=viz_config.get('plot_style', 'default'))
    
    # 收敛曲线
    visualizer.plot_convergence_curves(
        training_results,
        figsize=tuple(viz_config['figsize']['convergence'])
    )
    
    # 误差分量分解
    visualizer.plot_component_breakdown(
        training_results,
        figsize=tuple(viz_config['figsize']['component'])
    )
    
    # 泛化能力对比（如果有测试结果）
    if test_results and any(v is not None for v in test_results.values()):
        visualizer.plot_generalization_comparison(
            training_results,
            test_results,
            figsize=tuple(viz_config['figsize']['generalization'])
        )


def generate_report(
    config: dict,
    training_results: dict,
    test_results: dict
):
    """生成分析报告"""
    if not config['report']['generate_report']:
        print("\n跳过报告生成（配置中disabled）")
        return
    
    print("\n" + "="*80)
    print("四、生成分析报告")
    print("="*80 + "\n")
    
    output_dir = config['paths']['output_dir']
    report_config = config['report']
    
    generator = ReportGenerator(output_dir)
    
    report_path = generator.generate_complete_report(
        experiment_config=config['experiment'],
        training_results=training_results,
        test_results=test_results if test_results else None,
        output_filename=report_config['output_filename']
    )
    
    print(f"\n✓ 报告已生成: {report_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="BEVCalib实验分析统一工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--config',
        default='experiment_config.yaml',
        help='配置文件路径 (默认: experiment_config.yaml)'
    )
    parser.add_argument(
        '--skip-test',
        action='store_true',
        help='跳过测试集评估（但会尝试读取已有结果）'
    )
    parser.add_argument(
        '--only-train',
        action='store_true',
        help='只分析训练数据，不评估测试集'
    )
    
    args = parser.parse_args()
    
    # 加载配置
    print("="*80)
    print("BEVCalib 实验分析工具")
    print("="*80)
    print(f"\n配置文件: {args.config}\n")
    
    config = load_config(args.config)
    
    # 创建输出目录
    os.makedirs(config['paths']['output_dir'], exist_ok=True)
    
    # 1. 分析训练数据
    training_results = analyze_training(config)
    
    if not training_results:
        print("\n✗ 没有找到任何训练数据，退出")
        sys.exit(1)
    
    # 2. 评估测试数据
    test_results = {}
    if args.only_train:
        print("\n跳过测试集评估（--only-train）")
    elif config['evaluation'].get('run_test_eval', True) and not args.skip_test:
        test_results = evaluate_test_data(config, training_results)
    else:
        print("\n" + "="*80)
        print("二、尝试读取已有测试结果")
        print("="*80 + "\n")
        
        # 尝试读取已有的测试结果
        evaluator = TestEvaluator(
            config['paths']['bevcalib_root'],
            config['paths']['test_data']
        )
        
        for exp_config in config['experiments']:
            name = exp_config['name']
            if name not in training_results:
                continue
            
            model_dir = exp_config['model_dir']
            output_dir = os.path.join(
                config['paths']['logs_base'],
                model_dir,
                "test_data_eval"
            )
            
            if os.path.exists(os.path.join(output_dir, "extrinsics_and_errors.txt")):
                print(f"{name}: 读取已有结果...")
                test_results[name] = evaluator.parse_evaluation_results(output_dir)
                if test_results[name]:
                    print(f"  ✓ Total Rot: {test_results[name]['total']['mean']:.3f}° "
                          f"± {test_results[name]['total']['std']:.3f}°")
            else:
                print(f"{name}: 无测试结果")
                test_results[name] = None
    
    # 3. 生成可视化
    generate_visualizations(config, training_results, test_results)
    
    # 4. 生成报告
    generate_report(config, training_results, test_results)
    
    # 完成
    print("\n" + "="*80)
    print("✓ 分析完成！")
    print("="*80)
    print(f"\n结果保存在: {config['paths']['output_dir']}")
    print("\n生成的文件:")
    print("  • ANALYSIS_REPORT.md - 完整分析报告")
    print("  • convergence_curves.png - 收敛曲线")
    print("  • component_breakdown.png - 误差分量分解")
    if test_results and any(v is not None for v in test_results.values()):
        print("  • generalization_comparison.png - 泛化能力对比")
    print("")


if __name__ == "__main__":
    main()
