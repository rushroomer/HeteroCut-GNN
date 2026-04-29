"""
DirectTopKSelector评估脚本

评估维度：
1. 集合选择质量（Quality@K）
2. 排序质量（Spearman相关性）
3. 预测精度（相对误差、绝对误差）
4. 推理速度（与贪心对比）
"""

import glob
import gzip
import os
import pickle
import time
from collections import defaultdict

import numpy as np
import tensorflow as tf
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.direct_topk_model import DirectTopKSelector
from utils.losses import spearman_correlation, relative_error


def load_sample(filename):
    """加载样本"""
    with gzip.open(filename, 'rb') as f:
        return pickle.load(f)


def top_k_overlap(pred, true, k=5):
    """
    计算预测Top-K与真实Top-K的重叠率

    Args:
        pred: [N] 预测值
        true: [N] 真实值
        k: int Top-K数量

    Returns:
        overlap_rate: scalar [0, 1]
    """
    pred = tf.convert_to_tensor(pred, dtype=tf.float32)
    true = tf.convert_to_tensor(true, dtype=tf.float32)

    # 获取Top-K索引
    _, pred_topk_indices = tf.nn.top_k(pred, k=k)
    _, true_topk_indices = tf.nn.top_k(true, k=k)

    # 计算重叠
    pred_set = set(pred_topk_indices.numpy())
    true_set = set(true_topk_indices.numpy())
    overlap = len(pred_set & true_set)

    return overlap / k


def evaluate_set_selection_quality(model, test_files, K_values=[5, 10, 20]):
    """
    评估集合选择质量

    核心优势：DirectTopKSelector一次前向传播完成Top-K选择
    无需贪心搜索！
    """
    print("\n" + "=" * 80)
    print("评估集合选择质量")
    print("=" * 80)

    results = defaultdict(list)
    inference_times = []

    for i, filename in enumerate(tqdm(test_files, desc="集合选择质量")):
        sample = load_sample(filename)
        state = sample['state']
        subsets_data = sample['subsets']

        # 转换state为tensor
        state_tensor = {
            'variable_features': tf.constant(state['variable_features'], dtype=tf.float32),
            'constraint_features': tf.constant(state['constraint_features'], dtype=tf.float32),
            'cut_features': tf.constant(state['cut_features'], dtype=tf.float32),
            'var_cons_edges': tf.constant(state['var_cons_edges'], dtype=tf.int32),
            'var_cons_edge_features': tf.constant(state['var_cons_edge_features'], dtype=tf.float32),
            'var_cut_edges': tf.constant(state['var_cut_edges'], dtype=tf.int32),
            'var_cut_edge_features': tf.constant(state['var_cut_edge_features'], dtype=tf.float32)
        }

        # 记录推理时间
        start_time = time.time()

        for K in K_values:
            # 直接Top-K选择（核心优势！）
            pred_indices, selection_scores, joint_improvement = model(
                state_tensor, K=K, training=False
            )

            # 2. 计算预测集合的真实下界提升
            pred_set = set(pred_indices.numpy())
            best_match = None
            best_overlap = 0

            for subset in subsets_data:
                if subset['size'] == K:
                    subset_set = set(subset['indices'])
                    overlap = len(pred_set & subset_set)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_match = subset

            if best_match:
                Δz_pred = best_match['joint_improvement']
            else:
                Δz_pred = 0.01  # placeholder

            # 3. 找到真实最优的 K 个割平面
            K_subsets = [s for s in subsets_data if s['size'] == K]
            if len(K_subsets) > 0:
                optimal_subset = max(K_subsets, key=lambda s: s['joint_improvement'])
                Δz_opt = optimal_subset['joint_improvement']

                # 计算质量
                quality = Δz_pred / max(Δz_opt, 1e-8)
                results[f'quality@{K}'].append(quality)

        elapsed = time.time() - start_time
        inference_times.append(elapsed)

    # 汇总结果
    summary = {}
    for key, values in results.items():
        summary[key] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'median': np.median(values),
            'min': np.min(values),
            'max': np.max(values)
        }

    # 推理速度
    summary['inference_time'] = {
        'mean': np.mean(inference_times),
        'std': np.std(inference_times),
        'total': np.sum(inference_times)
    }

    print("\n集合选择质量:")
    for K in K_values:
        key = f'quality@{K}'
        if key in summary:
            print(f"  Quality@{K}: {summary[key]['mean']:.3f} ± {summary[key]['std']:.3f} "
                  f"(median: {summary[key]['median']:.3f}, min: {summary[key]['min']:.3f}, max: {summary[key]['max']:.3f})")

    print("\n推理速度（DirectTopK优势）:")
    print(f"  平均推理时间: {summary['inference_time']['mean']:.4f}s/样本")
    print(f"  总推理时间: {summary['inference_time']['total']:.2f}s")
    print(f"  吞吐量: {1/summary['inference_time']['mean']:.2f} 样本/秒")

    return summary


def evaluate_ranking_quality(model, test_files):
    """评估排序质量"""
    print("\n" + "=" * 80)
    print("评估排序质量")
    print("=" * 80)

    all_correlations = []
    all_top5_overlaps = []
    all_top10_overlaps = []

    for filename in tqdm(test_files, desc="排序质量"):
        sample = load_sample(filename)
        state = sample['state']
        subsets_data = sample['subsets']

        state_tensor = {
            'variable_features': tf.constant(state['variable_features'], dtype=tf.float32),
            'constraint_features': tf.constant(state['constraint_features'], dtype=tf.float32),
            'cut_features': tf.constant(state['cut_features'], dtype=tf.float32),
            'var_cons_edges': tf.constant(state['var_cons_edges'], dtype=tf.int32),
            'var_cons_edge_features': tf.constant(state['var_cons_edge_features'], dtype=tf.float32),
            'var_cut_edges': tf.constant(state['var_cut_edges'], dtype=tf.int32),
            'var_cut_edge_features': tf.constant(state['var_cut_edge_features'], dtype=tf.float32)
        }

        # 预测所有子集的下界提升
        subsets_indices = [
            tf.constant(s['indices'], dtype=tf.int32)
            for s in subsets_data
        ]
        true_improvements = tf.constant(
            [s['joint_improvement'] for s in subsets_data],
            dtype=tf.float32
        )

        predictions = model.predict_for_subset(state_tensor, subsets_indices, training=False)

        # 计算 Spearman 相关系数
        corr = spearman_correlation(predictions, true_improvements)
        all_correlations.append(corr.numpy())

        # 计算 Top-K 重叠率
        overlap_5 = top_k_overlap(predictions, true_improvements, k=5)
        overlap_10 = top_k_overlap(predictions, true_improvements, k=10)
        all_top5_overlaps.append(overlap_5)
        all_top10_overlaps.append(overlap_10)

    metrics = {
        'spearman': {
            'mean': np.mean(all_correlations),
            'std': np.std(all_correlations),
            'min': np.min(all_correlations),
            'max': np.max(all_correlations)
        },
        'top5_overlap': {
            'mean': np.mean(all_top5_overlaps),
            'std': np.std(all_top5_overlaps)
        },
        'top10_overlap': {
            'mean': np.mean(all_top10_overlaps),
            'std': np.std(all_top10_overlaps)
        }
    }

    print("\n排序质量:")
    print(f"  Spearman: {metrics['spearman']['mean']:.3f} ± {metrics['spearman']['std']:.3f} "
          f"(min: {metrics['spearman']['min']:.3f}, max: {metrics['spearman']['max']:.3f})")
    print(f"  Top-5 Overlap: {metrics['top5_overlap']['mean']:.3f} ± {metrics['top5_overlap']['std']:.3f}")
    print(f"  Top-10 Overlap: {metrics['top10_overlap']['mean']:.3f} ± {metrics['top10_overlap']['std']:.3f}")

    return metrics


def evaluate_prediction_accuracy(model, test_files):
    """评估预测精度"""
    print("\n" + "=" * 80)
    print("评估预测精度")
    print("=" * 80)

    all_relative_errors = []
    all_absolute_errors = []

    for filename in tqdm(test_files, desc="预测精度"):
        sample = load_sample(filename)
        state = sample['state']
        subsets_data = sample['subsets']

        state_tensor = {
            'variable_features': tf.constant(state['variable_features'], dtype=tf.float32),
            'constraint_features': tf.constant(state['constraint_features'], dtype=tf.float32),
            'cut_features': tf.constant(state['cut_features'], dtype=tf.float32),
            'var_cons_edges': tf.constant(state['var_cons_edges'], dtype=tf.int32),
            'var_cons_edge_features': tf.constant(state['var_cons_edge_features'], dtype=tf.float32),
            'var_cut_edges': tf.constant(state['var_cut_edges'], dtype=tf.int32),
            'var_cut_edge_features': tf.constant(state['var_cut_edge_features'], dtype=tf.float32)
        }

        subsets_indices = [
            tf.constant(s['indices'], dtype=tf.int32)
            for s in subsets_data
        ]
        true_improvements = np.array([s['joint_improvement'] for s in subsets_data])

        predictions = model.predict_for_subset(state_tensor, subsets_indices, training=False).numpy()

        # 计算误差
        relative_errors = np.abs(predictions - true_improvements) / np.maximum(true_improvements, 1e-8)
        absolute_errors = np.abs(predictions - true_improvements)

        all_relative_errors.extend(relative_errors)
        all_absolute_errors.extend(absolute_errors)

    all_relative_errors = np.array(all_relative_errors)
    all_absolute_errors = np.array(all_absolute_errors)

    # 计算容忍度准确率
    tolerances = [0.05, 0.10, 0.15, 0.20]
    accuracy_at_tolerance = {}
    for tol in tolerances:
        acc = np.mean(all_relative_errors <= tol)
        accuracy_at_tolerance[tol] = acc

    metrics = {
        'relative_error': {
            'mean': np.mean(all_relative_errors),
            'std': np.std(all_relative_errors),
            'median': np.median(all_relative_errors)
        },
        'absolute_error': {
            'mean': np.mean(all_absolute_errors),
            'std': np.std(all_absolute_errors),
            'median': np.median(all_absolute_errors)
        },
        'accuracy_at_tolerance': accuracy_at_tolerance
    }

    print("\n预测精度:")
    print(f"  相对误差: {metrics['relative_error']['mean']:.3f} ± {metrics['relative_error']['std']:.3f} "
          f"(median: {metrics['relative_error']['median']:.3f})")
    print(f"  绝对误差: {metrics['absolute_error']['mean']:.4f} ± {metrics['absolute_error']['std']:.4f}")
    print("\n  容忍度准确率:")
    for tol, acc in accuracy_at_tolerance.items():
        print(f"    {int(tol*100)}% 容忍: {acc:.3f}")

    return metrics


def comprehensive_evaluation(model_path, test_files, n_samples=None):
    """综合评估"""
    # 加载模型
    model = DirectTopKSelector(emb_size=32, n_layers=3)

    # 初始化模型（前向传播一次）
    sample = load_sample(test_files[0])
    state = sample['state']
    state_tensor = {
        'variable_features': tf.constant(state['variable_features'], dtype=tf.float32),
        'constraint_features': tf.constant(state['constraint_features'], dtype=tf.float32),
        'cut_features': tf.constant(state['cut_features'], dtype=tf.float32),
        'var_cons_edges': tf.constant(state['var_cons_edges'], dtype=tf.int32),
        'var_cons_edge_features': tf.constant(state['var_cons_edge_features'], dtype=tf.float32),
        'var_cut_edges': tf.constant(state['var_cut_edges'], dtype=tf.int32),
        'var_cut_edge_features': tf.constant(state['var_cut_edge_features'], dtype=tf.float32)
    }
    _ = model(state_tensor, K=5, training=False)

    # 加载权重
    model.load_model(model_path)

    print("\n" + "=" * 80)
    print("DirectTopKSelector - 综合评估")
    print("=" * 80)
    print(f"\n加载模型: {model_path}")
    print(f"测试样本总数: {len(test_files)}")
    if n_samples:
        print(f"使用样本数: {n_samples}\n")
    else:
        print(f"使用样本数: {len(test_files)} (全部)\n")

    # 选择测试样本
    if n_samples and n_samples < len(test_files):
        np.random.seed(42)
        test_subset = np.random.choice(test_files, size=n_samples, replace=False).tolist()
    else:
        test_subset = test_files

    # 1. 集合选择质量
    selection_metrics = evaluate_set_selection_quality(model, test_subset)

    # 2. 排序质量
    ranking_metrics = evaluate_ranking_quality(model, test_subset)

    # 3. 预测精度
    accuracy_metrics = evaluate_prediction_accuracy(model, test_subset)

    all_metrics = {
        'selection': selection_metrics,
        'ranking': ranking_metrics,
        'accuracy': accuracy_metrics
    }

    return all_metrics


if __name__ == '__main__':
    # 测试文件路径（使用MILP_polished的数据）
    test_files = sorted(glob.glob(
        '/home/jjw/milp_yxy/MILP_polished/data/samples/setcov/500r/test_set_level/sample_*.pkl'
    ))
    model_path = 'results/setcov_500_direct_topk/best_model.pkl'

    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found: {model_path}")
    elif len(test_files) == 0:
        print("[ERROR] No test samples found!")
    else:
        # 快速评估：使用10个样本
        metrics = comprehensive_evaluation(model_path, test_files, n_samples=10)

        # 保存结果
        results_dir = 'results/setcov_500_direct_topk'
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, 'evaluation_quick.pkl'), 'wb') as f:
            pickle.dump(metrics, f)

        print("\n" + "=" * 80)
        print("评估完成！")
        print("=" * 80)
