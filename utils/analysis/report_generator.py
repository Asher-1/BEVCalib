#!/usr/bin/env python3
"""
报告生成模块
生成Feishu兼容的Markdown分析报告
"""
import os
from typing import Dict, Optional, List
from datetime import datetime


class ReportGenerator:
    """报告生成器"""
    
    def __init__(self, output_dir: str):
        """
        初始化报告生成器
        
        Args:
            output_dir: 报告输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_complete_report(
        self,
        experiment_config: Dict,
        training_results: Dict[str, Dict],
        test_results: Optional[Dict[str, Dict]] = None,
        output_filename: str = "ANALYSIS_REPORT.md"
    ) -> str:
        """
        生成完整分析报告
        
        Args:
            experiment_config: 实验配置信息
            training_results: 训练结果数据
            test_results: 测试结果数据（可选）
            output_filename: 输出文件名
        
        Returns:
            报告文件路径
        """
        lines = []
        
        # 标题和配置
        exp_name = experiment_config.get('name', 'BEVCalib实验分析')
        lines.append(f"{exp_name}")
        lines.append("=" * 80)
        lines.append("")
        
        # 实验配置信息
        lines.append(f"**实验时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**实验组数**: {len(training_results)}")
        if 'dataset' in experiment_config:
            lines.append(f"**训练数据集**: {experiment_config['dataset']}")
        if 'test_dataset' in experiment_config:
            lines.append(f"**测试数据集**: {experiment_config['test_dataset']}")
        lines.append("")
        
        # 一、训练性能总结
        lines.append("=" * 80)
        lines.append("一、训练性能总结")
        lines.append("=" * 80)
        lines.append("")
        
        lines.extend(self._generate_training_summary_table(training_results))
        lines.append("")
        
        # 二、关键发现
        lines.append("1.1 关键发现")
        lines.append("")
        lines.extend(self._generate_key_findings(training_results))
        lines.append("")
        
        # 三、泛化能力评估
        if test_results and any(v is not None for v in test_results.values()):
            lines.append("=" * 80)
            lines.append("二、泛化能力评估")
            lines.append("=" * 80)
            lines.append("")
            lines.extend(self._generate_test_summary_table(training_results, test_results))
            lines.append("")
            lines.extend(self._generate_generalization_analysis(training_results, test_results))
            lines.append("")
        
        # 四、可视化图表
        lines.append("=" * 80)
        lines.append("三、可视化图表")
        lines.append("=" * 80)
        lines.append("")
        lines.append("![收敛曲线](convergence_curves.png)")
        lines.append("")
        lines.append("图1: 训练收敛曲线对比")
        lines.append("")
        lines.append("![误差分量](component_breakdown.png)")
        lines.append("")
        lines.append("图2: 误差分量详细分解")
        lines.append("")
        
        if test_results and any(v is not None for v in test_results.values()):
            lines.append("![泛化能力](generalization_comparison.png)")
            lines.append("")
            lines.append("图3: 泛化能力对比分析")
            lines.append("")
        
        # 五、结论与建议
        lines.append("=" * 80)
        lines.append("四、结论与建议")
        lines.append("=" * 80)
        lines.append("")
        lines.extend(self._generate_conclusions(training_results, test_results))
        lines.append("")
        
        # 保存报告
        report_text = "\n".join(lines)
        output_path = os.path.join(self.output_dir, output_filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"✓ Saved report: {output_path}")
        return output_path
    
    def _generate_training_summary_table(self, training_results: Dict) -> List[str]:
        """生成训练性能总结表格"""
        lines = []
        lines.append("1.1 最终训练精度")
        lines.append("")
        lines.append("| 实验 | Trans(m) | Fwd(m) | Lat(m) | Ht(m) | Rot(°) | Roll(°) | Pitch(°) | Yaw(°) |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        
        for name, data in training_results.items():
            final = data.get('final_metrics', {})
            if final:
                lines.append(
                    f"| {name} | "
                    f"{final.get('trans_error', 0):.4f} | "
                    f"{final.get('fwd', 0):.4f} | "
                    f"{final.get('lat', 0):.4f} | "
                    f"{final.get('ht', 0):.4f} | "
                    f"{final.get('rot_error', 0):.2f} | "
                    f"{final.get('roll', 0):.2f} | "
                    f"{final.get('pitch', 0):.2f} | "
                    f"{final.get('yaw', 0):.2f} |"
                )
        
        return lines
    
    def _generate_key_findings(self, training_results: Dict) -> List[str]:
        """生成关键发现"""
        lines = []
        
        # 过滤掉None结果
        valid_results = {k: v for k, v in training_results.items() 
                        if v is not None and v.get('final_metrics') is not None}
        
        if not valid_results:
            lines.append("⚠ 无有效训练数据")
            return lines
        
        # 找出最佳性能
        best_trans = min(valid_results.items(), 
                        key=lambda x: x[1].get('final_metrics', {}).get('trans_error', float('inf')))
        best_lat = min(valid_results.items(),
                      key=lambda x: x[1].get('final_metrics', {}).get('lat', float('inf')))
        best_rot = min(valid_results.items(),
                      key=lambda x: x[1].get('final_metrics', {}).get('rot_error', float('inf')))
        
        trans_err = best_trans[1].get('final_metrics', {}).get('trans_error', 0)
        lat_err = best_lat[1].get('final_metrics', {}).get('lat', 0)
        rot_err = best_rot[1].get('final_metrics', {}).get('rot_error', 0)
        
        if trans_err and trans_err > 0:
            lines.append(f"**最佳平移误差**: {best_trans[0]} (Trans: {trans_err:.4f}m)")
        if lat_err and lat_err > 0:
            lines.append(f"**最佳横向误差**: {best_lat[0]} (Lat: {lat_err:.4f}m)")
        lines.append(f"**最佳旋转误差**: {best_rot[0]} (Rot: {rot_err:.2f}°)")
        
        # 计算改进百分比
        if len(valid_results) >= 2:
            exp_list = list(valid_results.items())
            baseline = exp_list[0]
            best_item = exp_list[-1]
            
            # 尝试使用横向误差，如果没有则使用旋转误差
            baseline_lat = baseline[1].get('final_metrics', {}).get('lat', 0)
            best_lat_val = best_item[1].get('final_metrics', {}).get('lat', 0)
            
            if baseline_lat and baseline_lat > 0 and best_lat_val:
                improvement = (baseline_lat - best_lat_val) / baseline_lat * 100
                lines.append("")
                lines.append(f"**性能提升**: {best_item[0]}相比{baseline[0]}的横向误差改进 **{improvement:.1f}%**")
            else:
                # 使用旋转误差
                baseline_rot = baseline[1].get('final_metrics', {}).get('rot_error', 0)
                best_rot_val = best_item[1].get('final_metrics', {}).get('rot_error', 0)
                
                if baseline_rot and baseline_rot > 0 and best_rot_val:
                    improvement = (baseline_rot - best_rot_val) / baseline_rot * 100
                    lines.append("")
                    lines.append(f"**性能提升**: {best_item[0]}相比{baseline[0]}的旋转误差改进 **{improvement:.1f}%**")
        
        return lines
    
    def _generate_test_summary_table(
        self,
        training_results: Dict,
        test_results: Dict
    ) -> List[str]:
        """生成测试结果总结表格"""
        lines = []
        lines.append("2.1 测试集性能总结")
        lines.append("")
        lines.append("| 实验 | 训练误差(°) | 测试误差(°) | 泛化衰退 | 评估 |")
        lines.append("| --- | ---: | ---: | ---: | --- |")
        
        for name in training_results.keys():
            train_rot = training_results[name].get('final_metrics', {}).get('rot_error', 0)
            
            if name in test_results and test_results[name] is not None:
                test_rot = test_results[name]['total']['mean']
                test_std = test_results[name]['total']['std']
                degradation = test_rot / train_rot if train_rot > 0 else 0
                
                if degradation < 5:
                    status = "✓优秀"
                elif degradation < 7:
                    status = "✓良好"
                else:
                    status = "⚠需改进"
                
                lines.append(
                    f"| {name} | {train_rot:.2f} | "
                    f"{test_rot:.3f} ± {test_std:.3f} | "
                    f"{degradation:.2f}x | {status} |"
                )
            else:
                lines.append(f"| {name} | {train_rot:.2f} | - | - | 未评估 |")
        
        return lines
    
    def _generate_generalization_analysis(
        self,
        training_results: Dict,
        test_results: Dict
    ) -> List[str]:
        """生成泛化能力分析"""
        lines = []
        lines.append("2.2 泛化能力分析")
        lines.append("")
        
        # 找出泛化最好的模型
        valid_results = {
            name: test_results[name]['total']['mean'] / training_results[name].get('final_metrics', {}).get('rot_error', 1)
            for name in training_results.keys()
            if name in test_results and test_results[name] is not None
            and training_results[name].get('final_metrics', {}).get('rot_error', 0) > 0
        }
        
        if valid_results:
            best_gen = min(valid_results.items(), key=lambda x: x[1])
            lines.append(f"**最佳泛化模型**: {best_gen[0]} (衰退倍数: {best_gen[1]:.2f}x)")
            lines.append("")
            lines.append("泛化能力评估标准:")
            lines.append("  • <5x: 优秀 - 模型具有很强的泛化能力")
            lines.append("  • 5-7x: 良好 - 模型泛化能力可接受")
            lines.append("  • >7x: 需改进 - 可能存在过拟合")
        
        return lines
    
    def _generate_conclusions(
        self,
        training_results: Dict,
        test_results: Optional[Dict]
    ) -> List[str]:
        """生成结论与建议"""
        lines = []
        
        # 过滤有效结果
        valid_results = {k: v for k, v in training_results.items() 
                        if v is not None and v.get('final_metrics') is not None}
        
        if not valid_results:
            lines.append("⚠ 无有效训练数据")
            return lines
        
        # 训练性能结论 - 优先使用trans_error，否则使用rot_error
        trans_errors = {k: v.get('final_metrics', {}).get('trans_error', None) 
                       for k, v in valid_results.items()}
        trans_errors = {k: v for k, v in trans_errors.items() if v is not None and v > 0}
        
        if trans_errors:
            best_train = min(trans_errors.items(), key=lambda x: x[1])
            best_train = (best_train[0], valid_results[best_train[0]])
        else:
            # rotation-only模式，使用rot_error
            best_train = min(valid_results.items(),
                            key=lambda x: x[1].get('final_metrics', {}).get('rot_error', float('inf')))
        
        lines.append("4.1 核心结论")
        lines.append("")
        lines.append(f"1. **最佳训练精度**: {best_train[0]}")
        
        final_metrics = best_train[1].get('final_metrics', {})
        trans_err = final_metrics.get('trans_error', 0)
        rot_err = final_metrics.get('rot_error', 0)
        
        if trans_err and trans_err > 0:
            lines.append(f"   - 平移误差: {trans_err:.4f}m")
        lines.append(f"   - 旋转误差: {rot_err:.2f}°")
        lines.append("")
        
        # 泛化性能结论
        if test_results and any(v is not None for v in test_results.values()):
            valid_gen = {
                name: test_results[name]['total']['mean'] / training_results[name].get('final_metrics', {}).get('rot_error', 1)
                for name in training_results.keys()
                if name in test_results and test_results[name] is not None
                and training_results[name].get('final_metrics', {}).get('rot_error', 0) > 0
            }
            
            if valid_gen:
                best_gen = min(valid_gen.items(), key=lambda x: x[1])
                lines.append(f"2. **最佳泛化能力**: {best_gen[0]}")
                lines.append(f"   - 泛化衰退: {best_gen[1]:.2f}x")
                lines.append(f"   - 测试误差: {test_results[best_gen[0]]['total']['mean']:.3f}° ± {test_results[best_gen[0]]['total']['std']:.3f}°")
                lines.append("")
        
        lines.append("4.2 应用建议")
        lines.append("")
        lines.append("根据实验结果，建议:")
        
        if test_results and any(v is not None for v in test_results.values()):
            valid_gen = {
                name: test_results[name]['total']['mean'] / training_results[name].get('final_metrics', {}).get('rot_error', 1)
                for name in training_results.keys()
                if name in test_results and test_results[name] is not None
                and training_results[name].get('final_metrics', {}).get('rot_error', 0) > 0
            }
            
            if valid_gen:
                best_gen_name = min(valid_gen.items(), key=lambda x: x[1])[0]
                lines.append(f"  • **生产环境**: 推荐 {best_gen_name} (最佳泛化能力)")
        
        best_train_name = best_train[0]
        lines.append(f"  • **离线标定**: 推荐 {best_train_name} (最高训练精度)")
        
        return lines
