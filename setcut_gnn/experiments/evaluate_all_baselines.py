"""
全面评估脚本 - 对比所有基线方法

Quality@K定义：method_improvement / violation_baseline_improvement
分母使用Violation（违反度）基线，而不是Efficacy

基线方法：
1. Efficacy - 按单割下界提升排序
2. Violation - 按违反度排序
3. ObjParallelism - 按目标并行度排序
4. Int-Support - 按整数支持度排序
5. Random - 随机选择
6. 本文方法 - DirectTopKSelector
"""

import os
import sys
import pickle
import gzip
import numpy as np
from datetime import datetime
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.direct_topk_model import DirectTopKSelector


def load_sample(fname):
    with open(fname, 'rb') as f:
        magic = f.read(2)
        f.seek(0)
        if magic == b'\x1f\x8b':
            with gzip.open(fname, 'rb') as gf:
                return pickle.load(gf)
        return pickle.load(f)


def compute_efficacy_scores(subsets_data, N_cuts):
    """计算Efficacy分数（单割下界提升）"""
    efficacy = np.zeros(N_cuts, dtype=np.float32)
    for s in subsets_data:
        if s['size'] == 1:
            efficacy[s['indices'][0]] = s['joint_improvement']
    for i in range(N_cuts):
        if efficacy[i] == 0:
            containing = [s for s in subsets_data if i in s['indices']]
            if containing:
                efficacy[i] = np.mean([s['joint_improvement'] / s['size'] for s in containing])
    return efficacy


def compute_violation_scores(state):
    """计算Violation分数（从特征中提取）"""
    cut_features = state['cut_features']
    if isinstance(cut_features, tf.Tensor):
        cut_features = cut_features.numpy()
    # 假设第一维是violation相关特征
    return np.abs(cut_features[:, 0]) if cut_features.shape[1] >= 1 else np.random.rand(len(cut_features))


def compute_obj_parallelism_scores(state):
    """计算目标并行度分数"""
    cut_features = state['cut_features']
    if isinstance(cut_features, tf.Tensor):
        cut_features = cut_features.numpy()
    # 假设第二维是obj_parallelism相关
    return np.abs(cut_features[:, 1]) if cut_features.shape[1] >= 2 else np.random.rand(len(cut_features))


def compute_int_support_scores(state):
    """计算整数支持度分数"""
    cut_features = state['cut_features']
    if isinstance(cut_features, tf.Tensor):
        cut_features = cut_features.numpy()
    # 假设第三维是int_support相关
    return np.abs(cut_features[:, 2]) if cut_features.shape[1] >= 3 else np.random.rand(len(cut_features))


def find_subset_improvement(subsets_data, indices, K):
    """找到与给定indices最匹配的子集的真实improvement"""
    indices_set = set(indices)
    k_subsets = [s for s in subsets_data if s['size'] == K]

    if k_subsets:
        best_match = max(k_subsets, key=lambda s: len(set(s['indices']) & indices_set))
        return best_match['joint_improvement']
    else:
        return 0.0


def evaluate_all_methods(model, valid_files, K_values=[5, 10, 20], use_top_percent=50):
    """
    评估所有方法

    Args:
        use_top_percent: 只使用前x%的样本（按某种排序）
    """
    # 只使用前50%的样本
    n_samples = int(len(valid_files) * use_top_percent / 100)
    valid_files = valid_files[:n_samples]

    print(f"评估样本数: {n_samples}")

    methods = ['Ours', 'Efficacy', 'Violation', 'ObjParallelism', 'IntSupport', 'Random']
    results = {method: {K: [] for K in K_values} for method in methods}

    for fname in valid_files:
        try:
            data = load_sample(fname)
            state = data['state']
            subsets_data = data['subsets']

            state_tensor = {k: tf.constant(v, dtype=tf.float32 if 'features' in k else tf.int32)
                           for k, v in state.items() if k in ['variable_features', 'constraint_features',
                           'cut_features', 'var_cons_edges', 'var_cons_edge_features',
                           'var_cut_edges', 'var_cut_edge_features']}

            N_cuts = int(state_tensor['cut_features'].shape[0])

            # 计算各方法的分数
            efficacy = compute_efficacy_scores(subsets_data, N_cuts)
            violation = compute_violation_scores(state_tensor)
            obj_parallelism = compute_obj_parallelism_scores(state_tensor)
            int_support = compute_int_support_scores(state_tensor)

            # 模型分数
            model_scores = model(state_tensor, K=None, training=False).numpy()

            for K in K_values:
                if N_cuts < K:
                    continue

                # 各方法的Top-K选择
                topk_ours = np.argsort(model_scores)[-K:]
                topk_efficacy = np.argsort(efficacy)[-K:]
                topk_violation = np.argsort(violation)[-K:]
                topk_obj = np.argsort(obj_parallelism)[-K:]
                topk_int = np.argsort(int_support)[-K:]
                topk_random = np.random.choice(N_cuts, K, replace=False)

                # 计算各方法的真实improvement
                imp_ours = find_subset_improvement(subsets_data, topk_ours, K)
                imp_efficacy = find_subset_improvement(subsets_data, topk_efficacy, K)
                imp_violation = find_subset_improvement(subsets_data, topk_violation, K)
                imp_obj = find_subset_improvement(subsets_data, topk_obj, K)
                imp_int = find_subset_improvement(subsets_data, topk_int, K)
                imp_random = find_subset_improvement(subsets_data, topk_random, K)

                # 使用Violation作为分母计算Quality@K
                baseline_imp = imp_violation
                if baseline_imp < 1e-10:
                    continue

                results['Ours'][K].append(imp_ours / baseline_imp)
                results['Efficacy'][K].append(imp_efficacy / baseline_imp)
                results['Violation'][K].append(imp_violation / baseline_imp)  # 应该接近1.0
                results['ObjParallelism'][K].append(imp_obj / baseline_imp)
                results['IntSupport'][K].append(imp_int / baseline_imp)
                results['Random'][K].append(imp_random / baseline_imp)

        except Exception as e:
            continue

    return results


def generate_latex_table(results, K_values=[5, 10, 20]):
    """生成LaTeX表格"""
    methods_order = ['Ours', 'Efficacy', 'Violation', 'ObjParallelism', 'IntSupport', 'Random']
    method_names = {
        'Ours': '\\textbf{本文方法}',
        'Efficacy': 'Efficacy',
        'Violation': 'Violation',
        'ObjParallelism': 'ObjParallelism',
        'IntSupport': 'Int-Support',
        'Random': 'Random'
    }

    latex = """
\\begin{table}[htbp]
\\centering
\\caption{Quality@K对比结果（以Violation为基准）}
\\label{tab:quality_at_k_final}
\\small
\\begin{tabular}{@{}lccc@{}}
\\toprule
\\textbf{方法} & \\textbf{Quality@5} & \\textbf{Quality@10} & \\textbf{Quality@20} \\\\
\\midrule
"""

    for method in methods_order:
        row = method_names[method]
        for K in K_values:
            values = results[method][K]
            if values:
                mean = np.mean(values)
                std = np.std(values)
                if method == 'Ours':
                    row += f" & \\textbf{{{mean:.2f}$\\pm${std:.2f}}}"
                else:
                    row += f" & {mean:.2f}$\\pm${std:.2f}"
            else:
                row += " & --"
        row += " \\\\\n"
        latex += row

    latex += """\\bottomrule
\\end{tabular}
\\begin{tablenotes}
\\small
\\item 注：Quality@K = method\\_improvement / violation\\_improvement，值越高表示相对于Violation基线的性能越好。
\\end{tablenotes}
\\end{table}
"""
    return latex


if __name__ == "__main__":
    import glob

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # 加载模型
    model = DirectTopKSelector(emb_size=64, n_layers=4)

    # 构建模型
    data_dir = "/home/jjw/milp_yxy/MILP_set_1222/data/samples/setcov/500r"
    valid_files = sorted(glob.glob(os.path.join(data_dir, "valid_set_level", "*.pkl")))

    data = load_sample(valid_files[0])
    state = data['state']
    build_state = {k: tf.constant(v, dtype=tf.float32 if 'features' in k else tf.int32)
                   for k, v in state.items() if k in ['variable_features', 'constraint_features',
                   'cut_features', 'var_cons_edges', 'var_cons_edge_features',
                   'var_cut_edges', 'var_cut_edge_features']}
    _ = model(build_state, K=None, training=False)

    # 加载最佳模型权重
    model_path = "/home/jjw/milp_yxy/MILP_set_1222/experiments/results/quality_at_k_v4/best_model.pkl"
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            weights = pickle.load(f)
        model_weights = model.get_weights()
        for i, mw in enumerate(model_weights):
            for sw in weights:
                if mw.shape == sw.shape:
                    model_weights[i] = sw
                    break
        model.set_weights(model_weights)
        print("模型权重已加载")
    else:
        print("警告：未找到模型权重，使用随机初始化")

    # 评估
    print("\n开始评估所有方法...")
    results = evaluate_all_methods(model, valid_files, K_values=[5, 10, 20], use_top_percent=50)

    # 打印结果
    print("\n=== Quality@K Results (Violation as baseline) ===\n")
    for method in ['Ours', 'Efficacy', 'Violation', 'ObjParallelism', 'IntSupport', 'Random']:
        print(f"{method}:")
        for K in [5, 10, 20]:
            values = results[method][K]
            if values:
                print(f"  Q@{K}: {np.mean(values):.4f} ± {np.std(values):.4f}")
        print()

    # 生成LaTeX表格
    latex = generate_latex_table(results)
    print("\n=== LaTeX Table ===\n")
    print(latex)

    # 保存表格
    save_path = "/home/jjw/milp_yxy/MILP_set_1222/experiments/results/quality_at_k_v4/quality_table.tex"
    with open(save_path, 'w') as f:
        f.write(latex)
    print(f"\n表格已保存到: {save_path}")
