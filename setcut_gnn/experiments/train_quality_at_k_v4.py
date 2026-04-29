"""
训练脚本 - Quality@K损失函数（修正版v4）

核心修正：
使用真实improvement计算Quality@K损失，而不是模型预测值！
这样模型无法通过预测虚高值来"作弊"。

训练目标：让模型选择的Top-K割平面具有高真实improvement
"""

import os
import sys
import pickle
import gzip
import numpy as np
from datetime import datetime, timedelta
from time import perf_counter
from tensorflow.keras.optimizers import Adam
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


def preprocess_subsets(subsets_data, N_cuts):
    """预处理子集数据，提取efficacy和按K分组的子集"""
    efficacy = np.zeros(N_cuts, dtype=np.float32)
    k_subsets = {}

    for subset in subsets_data:
        size = subset['size']
        if size == 1:
            idx = subset['indices'][0]
            efficacy[idx] = subset['joint_improvement']

        if size not in k_subsets:
            k_subsets[size] = []
        k_subsets[size].append(subset)

    # 估计缺失的efficacy
    for i in range(N_cuts):
        if efficacy[i] == 0:
            containing = [s for s in subsets_data if i in s['indices']]
            if containing:
                efficacy[i] = np.mean([s['joint_improvement'] / s['size'] for s in containing])

    return efficacy, k_subsets


def find_real_improvement(model_topk_indices, k_subsets, efficacy, K):
    """
    找到与模型选择最匹配的子集的真实improvement

    Args:
        model_topk_indices: numpy array, 模型选择的Top-K索引
        k_subsets: dict, 按size分组的子集
        efficacy: numpy array, 单割efficacy
        K: int, 选择数量

    Returns:
        improvement: float, 真实的联合下界提升
        overlap: float, 与最匹配子集的重叠率
    """
    model_set = set(model_topk_indices)

    if K in k_subsets:
        subsets_k = k_subsets[K]
        # 找与模型选择重叠最多的子集
        best_match = max(subsets_k, key=lambda s: len(set(s['indices']) & model_set))
        improvement = best_match['joint_improvement']
        overlap = len(set(best_match['indices']) & model_set) / K
    else:
        # 没有size=K的子集，用单割之和估计
        improvement = np.sum(efficacy[list(model_set)])
        overlap = 0.0

    return improvement, overlap


def find_efficacy_improvement(efficacy, k_subsets, K):
    """计算Efficacy Top-K的真实improvement"""
    efficacy_topk = np.argsort(efficacy)[-K:]
    efficacy_set = set(efficacy_topk)

    if K in k_subsets:
        subsets_k = k_subsets[K]
        best_match = max(subsets_k, key=lambda s: len(set(s['indices']) & efficacy_set))
        return best_match['joint_improvement'], efficacy_topk
    else:
        return np.sum(efficacy[efficacy_topk]), efficacy_topk


def compute_subset_target_scores(subsets_data, N_cuts, K):
    """
    计算每个割平面的目标分数，基于它在高质量子集中的出现频率

    核心思想：如果一个割平面频繁出现在高improvement的K-size子集中，
    它就应该有更高的分数
    """
    # 获取所有K-size子集
    k_subsets = [s for s in subsets_data if s['size'] == K]
    if not k_subsets:
        return None

    # 按improvement排序
    k_subsets_sorted = sorted(k_subsets, key=lambda s: s['joint_improvement'], reverse=True)

    # 统计每个割平面在top子集中的加权出现次数
    scores = np.zeros(N_cuts, dtype=np.float32)

    # 只使用top-50%的子集
    top_n = max(1, len(k_subsets_sorted) // 2)
    for rank, subset in enumerate(k_subsets_sorted[:top_n]):
        # 权重随排名递减
        weight = 1.0 / (rank + 1)
        for idx in subset['indices']:
            scores[idx] += weight * subset['joint_improvement']

    # 归一化
    if scores.max() > 0:
        scores = scores / scores.max()

    return scores


def train_step_real_quality(model, optimizer, state, subsets_data, efficacy, k_subsets,
                            K_values=[5, 10, 20], clip_norm=1.0):
    """
    训练步骤 - 使用真实improvement计算Quality@K

    核心思想：
    1. 从数据中学习哪些割平面组合效果好
    2. 让模型选择这些高质量组合中的割平面
    3. 这样可以超越efficacy（因为efficacy只看单个，不看组合效果）
    """
    N_cuts = int(state['cut_features'].shape[0])

    with tf.GradientTape() as tape:
        # 获取选择分数
        selection_scores = model(state, K=None, training=True)  # [N_cuts]

        total_loss = 0.0
        quality_results = {}

        for K in K_values:
            if N_cuts < K:
                continue

            # 模型选择Top-K（用于评估）
            model_topk_indices = tf.math.top_k(selection_scores, k=K).indices.numpy()

            # 找模型选择的真实improvement
            model_improvement, overlap = find_real_improvement(
                model_topk_indices, k_subsets, efficacy, K
            )

            # 找Efficacy的improvement
            efficacy_improvement, efficacy_topk = find_efficacy_improvement(efficacy, k_subsets, K)
            efficacy_improvement = max(efficacy_improvement, 1e-8)

            # 计算真实Quality@K
            quality_k = model_improvement / efficacy_improvement
            quality_results[K] = quality_k

            # 损失设计：让模型学习选择高质量子集中的割平面
            # 目标分数基于割平面在高质量子集中的出现频率
            target_scores = compute_subset_target_scores(subsets_data, N_cuts, K)

            if target_scores is not None:
                target_scores_tf = tf.constant(target_scores, dtype=tf.float32)

                # ListNet损失：概率分布匹配
                target_probs = tf.nn.softmax(target_scores_tf / 0.1)
                pred_probs = tf.nn.softmax(selection_scores / 0.1)

                # 交叉熵损失
                ce_loss = -tf.reduce_sum(target_probs * tf.math.log(pred_probs + 1e-10))

                # 加权
                if K == 5:
                    weight = 2.0
                elif K == 10:
                    weight = 1.0
                else:
                    weight = 0.5

                total_loss += weight * ce_loss

        # 归一化
        if total_loss > 0:
            total_loss = total_loss / 3.5

        # 添加值预测辅助损失（帮助模型理解子集质量）
        subset_sample = subsets_data[:30]
        if len(subset_sample) > 0:
            cut_indices_list = [tf.constant(s['indices'], dtype=tf.int32) for s in subset_sample]
            true_improvements = tf.constant([s['joint_improvement'] for s in subset_sample], dtype=tf.float32)

            predictions = model.predict_for_subset(state, cut_indices_list, training=True)

            # 相对误差损失
            value_loss = tf.reduce_mean(tf.abs(predictions - true_improvements) / (true_improvements + 1e-8))
            total_loss += 0.3 * value_loss

    # 计算梯度并裁剪
    gradients = tape.gradient(total_loss, model.trainable_variables)
    if gradients[0] is not None:
        gradients, _ = tf.clip_by_global_norm(gradients, clip_norm)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return total_loss.numpy(), quality_results


def evaluate_real_quality(model, valid_files, K_values=[5, 10, 20], max_samples=200):
    """评估真实Quality@K"""
    quality_results = {K: [] for K in K_values}

    for fname in valid_files[:max_samples]:
        try:
            data = load_sample(fname)
            state = data['state']
            subsets_data = data['subsets']

            state_tensor = {k: tf.constant(v, dtype=tf.float32 if 'features' in k else tf.int32)
                           for k, v in state.items() if k in ['variable_features', 'constraint_features',
                           'cut_features', 'var_cons_edges', 'var_cons_edge_features',
                           'var_cut_edges', 'var_cut_edge_features']}

            N_cuts = int(state_tensor['cut_features'].shape[0])
            efficacy, k_subsets = preprocess_subsets(subsets_data, N_cuts)

            # 模型选择
            selection_scores = model(state_tensor, K=None, training=False).numpy()

            for K in K_values:
                if N_cuts < K:
                    continue

                model_topk = np.argsort(selection_scores)[-K:]
                model_improvement, _ = find_real_improvement(model_topk, k_subsets, efficacy, K)
                efficacy_improvement, _ = find_efficacy_improvement(efficacy, k_subsets, K)

                if efficacy_improvement > 1e-8:
                    qk = model_improvement / efficacy_improvement
                    if qk < 50:  # 过滤异常值
                        quality_results[K].append(qk)
        except:
            continue

    return {K: np.mean(v) if v else 0.0 for K, v in quality_results.items()}


def train_model(train_files, valid_files, save_dir,
                emb_size=64, n_layers=4, max_epochs=100,
                lr=1e-4, clip_norm=1.0,
                K_values=[5, 10, 20],
                samples_per_epoch=100,  # 每个epoch的样本数
                patience=10, early_stopping=30):
    """训练模型 - 快速版本"""
    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, 'training.log')

    def log(msg):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        formatted = f"[{timestamp}] {msg}"
        print(formatted)
        with open(log_file, 'a') as f:
            f.write(formatted + '\n')

    log("=" * 80)
    log("DirectTopKSelector训练 - 真实Quality@K损失（快速版v4）")
    log("=" * 80)
    log(f"核心改进: 使用真实improvement计算Quality@K，防止模型作弊")
    log(f"训练样本数: {len(train_files)} (每epoch采样{samples_per_epoch}个)")
    log(f"验证样本数: {len(valid_files)}")
    log(f"嵌入维度: {emb_size}, 层数: {n_layers}")
    log(f"学习率: {lr}, 梯度裁剪: {clip_norm}")
    log("")

    # 创建模型
    model = DirectTopKSelector(emb_size=emb_size, n_layers=n_layers)
    optimizer = Adam(learning_rate=lr)

    # 构建模型
    data = load_sample(train_files[0])
    state = data['state']
    build_state = {k: tf.constant(v, dtype=tf.float32 if 'features' in k else tf.int32)
                   for k, v in state.items() if k in ['variable_features', 'constraint_features',
                   'cut_features', 'var_cons_edges', 'var_cons_edge_features',
                   'var_cut_edges', 'var_cut_edge_features']}
    _ = model(build_state, K=None, training=False)
    log(f"模型已构建，权重数量: {len(model.get_weights())}")

    best_quality_score = 0.0
    plateau_count = 0
    no_improve_count = 0
    start_time = perf_counter()

    for epoch in range(1, max_epochs + 1):
        epoch_start = perf_counter()
        log(f"\n{'='*60}")
        log(f"[Epoch {epoch}/{max_epochs}]")
        log('='*60)

        # 训练 - 每epoch随机采样samples_per_epoch个样本
        train_losses = []
        train_qualities = {K: [] for K in K_values}

        # 随机采样
        epoch_files = np.random.choice(train_files, size=min(samples_per_epoch, len(train_files)), replace=False)

        for i, fname in enumerate(epoch_files):
            try:
                data = load_sample(fname)
                state = data['state']
                subsets_data = data['subsets']

                state_tensor = {k: tf.constant(v, dtype=tf.float32 if 'features' in k else tf.int32)
                               for k, v in state.items() if k in ['variable_features', 'constraint_features',
                               'cut_features', 'var_cons_edges', 'var_cons_edge_features',
                               'var_cut_edges', 'var_cut_edge_features']}

                N_cuts = int(state_tensor['cut_features'].shape[0])
                efficacy, k_subsets = preprocess_subsets(subsets_data, N_cuts)

                loss, quality_results = train_step_real_quality(
                    model, optimizer, state_tensor, subsets_data,
                    efficacy, k_subsets, K_values, clip_norm
                )

                train_losses.append(loss)
                for K in K_values:
                    if K in quality_results:
                        train_qualities[K].append(quality_results[K])

            except Exception as e:
                continue

        # 训练总结
        epoch_time = perf_counter() - epoch_start
        q5 = np.mean(train_qualities[5]) if train_qualities[5] else 0
        q10 = np.mean(train_qualities[10]) if train_qualities[10] else 0
        q20 = np.mean(train_qualities[20]) if train_qualities[20] else 0
        log(f"  [Train] Loss: {np.mean(train_losses):.4f}, "
            f"Q@5: {q5:.3f}, Q@10: {q10:.3f}, Q@20: {q20:.3f} ({epoch_time:.1f}s)")

        # 验证 - 使用真实Quality@K（只用50个样本加速）
        log("  [Validation]")
        avg_qualities = evaluate_real_quality(model, valid_files, K_values, max_samples=50)

        for K in K_values:
            status = "!" if avg_qualities[K] >= 1.0 else ""
            log(f"    Quality@{K}: {avg_qualities[K]:.4f} {status}")

        current_score = (2.0 * avg_qualities.get(5, 0) +
                        1.0 * avg_qualities.get(10, 0) +
                        0.5 * avg_qualities.get(20, 0)) / 3.5
        log(f"    综合分数: {current_score:.4f}")

        # 保存最佳模型
        if current_score > best_quality_score:
            best_quality_score = current_score
            plateau_count = 0
            no_improve_count = 0

            with open(os.path.join(save_dir, 'best_model.pkl'), 'wb') as f:
                pickle.dump(model.get_weights(), f)

            info = {'epoch': epoch, 'score': current_score, **avg_qualities}
            with open(os.path.join(save_dir, 'best_model_info.pkl'), 'wb') as f:
                pickle.dump(info, f)

            log(f"\n  *** 最佳模型已保存 (Score: {current_score:.4f}) ***")
        else:
            no_improve_count += 1
            plateau_count += 1

            if plateau_count >= patience:
                old_lr = optimizer.learning_rate.numpy()
                new_lr = old_lr * 0.5
                optimizer.learning_rate.assign(new_lr)
                log(f"\n  [学习率衰减] {old_lr:.2e} -> {new_lr:.2e}")
                plateau_count = 0

            if no_improve_count >= early_stopping:
                log(f"\n  [早停] {early_stopping} 轮无改进")
                break

    elapsed = perf_counter() - start_time
    log(f"\n训练完成，耗时: {str(timedelta(seconds=int(elapsed)))}")
    log(f"最佳分数: {best_quality_score:.4f}")

    return model


if __name__ == "__main__":
    import glob

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    data_dir = "/home/jjw/milp_yxy/MILP_set_1222/data/samples/setcov/500r"
    train_files = sorted(glob.glob(os.path.join(data_dir, "train_set_level", "*.pkl")))
    valid_files = sorted(glob.glob(os.path.join(data_dir, "valid_set_level", "*.pkl")))

    print(f"训练: {len(train_files)}, 验证: {len(valid_files)}")

    save_dir = "/home/jjw/milp_yxy/MILP_set_1222/experiments/results/quality_at_k_v4"

    model = train_model(
        train_files, valid_files, save_dir,
        emb_size=64, n_layers=4, max_epochs=100,
        lr=1e-3, clip_norm=1.0,  # 提高学习率
        K_values=[5, 10, 20],
        samples_per_epoch=100,  # 每epoch只用100个样本
        patience=10, early_stopping=30
    )

    print(f"\n模型保存在: {save_dir}/best_model.pkl")
