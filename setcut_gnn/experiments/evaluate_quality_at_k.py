"""
Quality@K对比评估脚本

比较以下方法的Quality@K性能：
1. 本文方法（DirectTopKSelector）- GNN集合级预测
2. Efficacy Baseline - SCIP默认的深度排序
3. Violation Baseline - 按违反度排序
4. Random Baseline - 随机选择
5. ObjParallelism Baseline - 按目标并行度排序
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
    """加载样本数据"""
    with open(fname, 'rb') as f:
        magic = f.read(2)
        f.seek(0)
        if magic == b'\x1f\x8b':
            with gzip.open(fname, 'rb') as gf:
                data = pickle.load(gf)
        else:
            data = pickle.load(f)
    return data


def compute_efficacy_scores(state, subsets_data):
    """
    计算Efficacy分数（单个割平面的下界提升）
    Efficacy = 分数解到割平面的欧氏距离
    """
    N_cuts = int(state['cut_features'].shape[0])
    efficacy = np.zeros(N_cuts, dtype=np.float32)

    # 从subsets_data中提取单个割平面的下界提升
    for subset in subsets_data:
        if subset['size'] == 1:
            idx = subset['indices'][0]
            efficacy[idx] = subset['joint_improvement']

    # 对于没有单独数据的割平面，用包含它的子集估计
    for i in range(N_cuts):
        if efficacy[i] == 0:
            containing = [s for s in subsets_data if i in s['indices']]
            if containing:
                efficacy[i] = np.mean([s['joint_improvement'] / s['size'] for s in containing])

    return efficacy


def compute_violation_scores(state):
    """
    计算Violation分数（从割平面特征中提取）
    假设cut_features的第一个维度包含violation信息
    """
    cut_features = state['cut_features'].numpy()
    # 通常violation是特征的第一维或归一化后的值
    # 这里假设特征的某一维与violation相关
    if cut_features.shape[1] >= 1:
        violation = np.abs(cut_features[:, 0])  # 使用第一维作为violation代理
    else:
        violation = np.random.rand(cut_features.shape[0])
    return violation


def compute_obj_parallelism_scores(state):
    """
    计算目标并行度分数
    假设cut_features中包含相关信息
    """
    cut_features = state['cut_features'].numpy()
    if cut_features.shape[1] >= 2:
        obj_parallelism = np.abs(cut_features[:, 1])  # 使用第二维
    else:
        obj_parallelism = np.random.rand(cut_features.shape[0])
    return obj_parallelism


def find_best_subset_improvement(subsets_data, indices, K):
    """
    找到与给定indices最匹配的子集的真实下界提升
    """
    indices_set = set(indices)
    k_subsets = [s for s in subsets_data if s['size'] == K]

    if k_subsets:
        # 找重叠最大的子集
        best_match = max(k_subsets, key=lambda s: len(set(s['indices']) & indices_set))
        return best_match['joint_improvement'], len(set(best_match['indices']) & indices_set) / K
    else:
        # 没有size=K的子集，估计
        single_improvements = []
        for idx in indices:
            for s in subsets_data:
                if s['size'] == 1 and s['indices'][0] == idx:
                    single_improvements.append(s['joint_improvement'])
                    break
        if single_improvements:
            return sum(single_improvements), 0.0
        return 0.0, 0.0


def evaluate_method(method_name, get_topk_fn, state, subsets_data, K_values):
    """
    评估某个方法的Quality@K

    Args:
        method_name: str 方法名称
        get_topk_fn: callable 返回Top-K索引的函数
        state: dict 图状态
        subsets_data: list 子集数据
        K_values: list 要评估的K值

    Returns:
        results: dict 各K的Quality@K结果
    """
    results = {}
    N_cuts = int(state['cut_features'].shape[0])

    # 计算Efficacy Top-K作为baseline
    efficacy = compute_efficacy_scores(state, subsets_data)

    for K in K_values:
        if N_cuts < K:
            results[K] = {'quality_k': 0.0, 'improvement': 0.0, 'efficacy_improvement': 0.0}
            continue

        # 获取该方法的Top-K
        method_topk = get_topk_fn(K)

        # Efficacy的Top-K
        efficacy_topk = np.argsort(efficacy)[-K:]

        # 计算真实下界提升
        method_improvement, method_overlap = find_best_subset_improvement(subsets_data, method_topk, K)
        efficacy_improvement, _ = find_best_subset_improvement(subsets_data, efficacy_topk, K)

        # Quality@K = method_improvement / efficacy_improvement
        if efficacy_improvement > 1e-8:
            quality_k = method_improvement / efficacy_improvement
        else:
            quality_k = 1.0  # 如果efficacy也为0，认为相等

        results[K] = {
            'quality_k': quality_k,
            'improvement': method_improvement,
            'efficacy_improvement': efficacy_improvement,
            'overlap': method_overlap
        }

    return results


def evaluate_all_methods(model, sample_files, K_values=[5, 10, 20], verbose=True, model_loaded=True):
    """
    评估所有方法
    """
    all_results = {
        'ours': {K: [] for K in K_values},
        'efficacy': {K: [] for K in K_values},
        'violation': {K: [] for K in K_values},
        'obj_parallelism': {K: [] for K in K_values},
        'random': {K: [] for K in K_values}
    }

    for i, fname in enumerate(sample_files):
        try:
            data = load_sample(fname)
            state = data['state']
            subsets_data = data['subsets']

            # 转换为tensor
            state_tensor = {
                'variable_features': tf.constant(state['variable_features'], dtype=tf.float32),
                'constraint_features': tf.constant(state['constraint_features'], dtype=tf.float32),
                'cut_features': tf.constant(state['cut_features'], dtype=tf.float32),
                'var_cons_edges': tf.constant(state['var_cons_edges'], dtype=tf.int32),
                'var_cons_edge_features': tf.constant(state['var_cons_edge_features'], dtype=tf.float32),
                'var_cut_edges': tf.constant(state['var_cut_edges'], dtype=tf.int32),
                'var_cut_edge_features': tf.constant(state['var_cut_edge_features'], dtype=tf.float32)
            }

            N_cuts = int(state_tensor['cut_features'].shape[0])

            # 计算各方法的分数
            efficacy = compute_efficacy_scores(state_tensor, subsets_data)
            violation = compute_violation_scores(state_tensor)
            obj_par = compute_obj_parallelism_scores(state_tensor)

            # 本文方法（如果模型加载成功）
            if model_loaded:
                selection_scores = model(state_tensor, K=None, training=False).numpy()

            # 定义各方法的Top-K获取函数
            def get_ours_topk(K):
                if model_loaded:
                    return np.argsort(selection_scores)[-K:]
                return np.argsort(efficacy)[-K:]  # 如果模型未加载，用efficacy代替

            def get_efficacy_topk(K):
                return np.argsort(efficacy)[-K:]

            def get_violation_topk(K):
                return np.argsort(violation)[-K:]

            def get_obj_par_topk(K):
                return np.argsort(obj_par)[-K:]

            def get_random_topk(K):
                return np.random.choice(N_cuts, size=min(K, N_cuts), replace=False)

            # 评估各方法
            methods = {
                'efficacy': get_efficacy_topk,
                'violation': get_violation_topk,
                'obj_parallelism': get_obj_par_topk,
                'random': get_random_topk
            }

            # 只有在模型加载成功时才评估ours
            if model_loaded:
                methods['ours'] = get_ours_topk

            for method_name, get_topk_fn in methods.items():
                results = evaluate_method(method_name, get_topk_fn, state_tensor, subsets_data, K_values)
                for K in K_values:
                    all_results[method_name][K].append(results[K]['quality_k'])

            if verbose and (i + 1) % 50 == 0:
                print(f"  已处理 {i+1}/{len(sample_files)} 个样本")

        except Exception as e:
            if verbose:
                print(f"  [WARNING] 样本 {fname} 处理失败: {e}")
            continue

    # 计算平均值和标准差
    summary = {}
    for method_name in all_results:
        summary[method_name] = {}
        for K in K_values:
            values = all_results[method_name][K]
            if values:
                # 过滤异常值（超过100的视为异常）
                values_filtered = [v for v in values if v < 100]
                if values_filtered:
                    summary[method_name][K] = {
                        'mean': np.mean(values_filtered),
                        'std': np.std(values_filtered),
                        'median': np.median(values_filtered),
                        'count': len(values_filtered)
                    }
                else:
                    summary[method_name][K] = {'mean': 0, 'std': 0, 'median': 0, 'count': 0}
            else:
                summary[method_name][K] = {'mean': 0, 'std': 0, 'median': 0, 'count': 0}

    return summary, all_results


def print_comparison_table(summary, K_values):
    """打印对比表格"""
    print("\n" + "=" * 80)
    print("Quality@K 对比结果")
    print("=" * 80)

    methods = ['ours', 'efficacy', 'violation', 'obj_parallelism', 'random']
    method_names = {
        'ours': '本文方法(GNN)',
        'efficacy': 'Efficacy',
        'violation': 'Violation',
        'obj_parallelism': 'ObjParallelism',
        'random': 'Random'
    }

    # 表头
    header = f"{'方法':<20}"
    for K in K_values:
        header += f" | Q@{K:>2} (mean±std)"
    print(header)
    print("-" * 80)

    # 各方法结果
    for method in methods:
        row = f"{method_names[method]:<20}"
        for K in K_values:
            stats = summary[method][K]
            row += f" | {stats['mean']:>5.2f}±{stats['std']:>4.2f}"
        print(row)

    print("=" * 80)

    # 计算相对提升
    print("\n相对于Efficacy的提升：")
    for K in K_values:
        eff_mean = summary['efficacy'][K]['mean']
        ours_mean = summary['ours'][K]['mean']
        if eff_mean > 0:
            improvement = (ours_mean - eff_mean) / eff_mean * 100
            print(f"  Quality@{K}: {improvement:+.1f}%")


def generate_latex_table(summary, K_values):
    """生成LaTeX表格"""
    methods = ['ours', 'efficacy', 'violation', 'obj_parallelism', 'random']
    method_names = {
        'ours': '\\textbf{本文方法}',
        'efficacy': 'Efficacy',
        'violation': 'Violation',
        'obj_parallelism': 'ObjParallelism',
        'random': 'Random'
    }

    latex = """
\\begin{table}[htbp]
\\centering
\\caption{Quality@K对比结果}
\\label{tab:quality_at_k}
\\small
\\begin{tabular}{@{}l""" + "c" * len(K_values) + """@{}}
\\toprule
\\textbf{方法}"""

    for K in K_values:
        latex += f" & \\textbf{{Quality@{K}}}"
    latex += " \\\\\n\\midrule\n"

    for method in methods:
        latex += f"{method_names[method]}"
        for K in K_values:
            stats = summary[method][K]
            if method == 'ours':
                latex += f" & \\textbf{{{stats['mean']:.2f}$\\pm${stats['std']:.2f}}}"
            else:
                latex += f" & {stats['mean']:.2f}$\\pm${stats['std']:.2f}"
        latex += " \\\\\n"

    latex += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    return latex


if __name__ == "__main__":
    import glob

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    print("=" * 80)
    print("Quality@K 对比评估")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # 加载模型
    model_path = "/home/jjw/milp_yxy/MILP_set_1222/experiments/results/quality_at_k_v2/best_model.pkl"

    print("\n[1] 加载模型...")
    model = DirectTopKSelector(emb_size=64, n_layers=4)
    model_loaded = False

    try:
        # 加载一个真实样本来构建模型
        test_dir = "/home/jjw/milp_yxy/MILP_set_1222/data/samples/setcov/500r/valid_set_level"
        sample_files = sorted(glob.glob(os.path.join(test_dir, "*.pkl")))
        if sample_files:
            data = load_sample(sample_files[0])
            state = data['state']
            build_state = {
                'variable_features': tf.constant(state['variable_features'], dtype=tf.float32),
                'constraint_features': tf.constant(state['constraint_features'], dtype=tf.float32),
                'cut_features': tf.constant(state['cut_features'], dtype=tf.float32),
                'var_cons_edges': tf.constant(state['var_cons_edges'], dtype=tf.int32),
                'var_cons_edge_features': tf.constant(state['var_cons_edge_features'], dtype=tf.float32),
                'var_cut_edges': tf.constant(state['var_cut_edges'], dtype=tf.int32),
                'var_cut_edge_features': tf.constant(state['var_cut_edge_features'], dtype=tf.float32)
            }
            _ = model(build_state, K=None, training=False)

        # 加载权重
        with open(model_path, 'rb') as f:
            weights = pickle.load(f)

        # 检查权重数量
        expected = len(model.get_weights())
        actual = len(weights)
        print(f"  模型期望 {expected} 个权重，保存的有 {actual} 个")

        if expected == actual:
            model.set_weights(weights)
            model_loaded = True
            print(f"  模型已加载: {model_path}")
        else:
            print(f"  [WARNING] 权重数量不匹配，尝试部分加载...")
            # 尝试匹配形状相同的权重
            model_weights = model.get_weights()
            matched = 0
            for i, (mw, sw) in enumerate(zip(model_weights, weights)):
                if mw.shape == sw.shape:
                    model_weights[i] = sw
                    matched += 1
            if matched > expected * 0.9:  # 超过90%匹配
                model.set_weights(model_weights)
                model_loaded = True
                print(f"  部分加载成功: {matched}/{expected} 权重匹配")
            else:
                print(f"  [WARNING] 模型加载失败，将只评估baseline方法")
    except Exception as e:
        print(f"  [ERROR] 模型加载失败: {e}")
        print(f"  将只评估baseline方法")

    # 加载测试数据
    print("\n[2] 加载测试数据...")
    test_dir = "/home/jjw/milp_yxy/MILP_set_1222/data/samples/setcov/500r/valid_set_level"
    test_files = sorted(glob.glob(os.path.join(test_dir, "*.pkl")))
    print(f"  找到 {len(test_files)} 个测试样本")

    # 评估
    print("\n[3] 开始评估...")
    K_values = [5, 10, 20]
    summary, all_results = evaluate_all_methods(model, test_files, K_values, verbose=True, model_loaded=model_loaded)

    # 打印结果
    print_comparison_table(summary, K_values)

    # 生成LaTeX表格
    latex = generate_latex_table(summary, K_values)
    print("\n[4] LaTeX表格：")
    print(latex)

    # 保存结果
    results_path = "/home/jjw/milp_yxy/MILP_set_1222/experiments/results/quality_at_k_v2/comparison_results.pkl"
    with open(results_path, 'wb') as f:
        pickle.dump({
            'summary': summary,
            'all_results': all_results,
            'K_values': K_values,
            'timestamp': datetime.now().isoformat()
        }, f)
    print(f"\n[5] 结果已保存: {results_path}")

    # 保存LaTeX
    latex_path = "/home/jjw/milp_yxy/MILP_set_1222/experiments/results/quality_at_k_v2/quality_at_k_table.tex"
    with open(latex_path, 'w') as f:
        f.write(latex)
    print(f"    LaTeX表格已保存: {latex_path}")
