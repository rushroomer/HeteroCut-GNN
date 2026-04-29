"""
损失函数 - DirectTopKSelector专用

包含两部分：
1. 选择损失（L_selection）：确保好的割平面得到高选择分数
2. 值预测损失（L_value）：准确预测联合下界提升
"""

import tensorflow as tf


def selection_ranking_loss(selection_scores, cut_qualities, margin=0.1):
    """
    选择排序损失 (Pairwise Hinge Loss)

    目标：质量高的割平面应该得到更高的选择分数

    Args:
        selection_scores: [N_cuts] 所有割平面的选择分数
        cut_qualities: [N_cuts] 每个割平面的质量（基于其在好子集中的出现频率）
        margin: float ranking margin

    Returns:
        loss: scalar
    """
    N = tf.shape(selection_scores)[0]

    # 展开为成对比较
    scores_i = tf.expand_dims(selection_scores, 1)  # [N, 1]
    scores_j = tf.expand_dims(selection_scores, 0)  # [1, N]
    score_diff = scores_i - scores_j  # [N, N]

    qualities_i = tf.expand_dims(cut_qualities, 1)
    qualities_j = tf.expand_dims(cut_qualities, 0)
    quality_diff = qualities_i - qualities_j  # [N, N]

    # Hinge loss: 如果质量i > 质量j，则希望分数i > 分数j + margin
    sign = tf.sign(quality_diff)
    pair_loss = tf.maximum(0.0, -sign * score_diff + margin)

    # 只统计质量明显不同的对
    valid_pairs = tf.cast(tf.abs(quality_diff) > 1e-6, tf.float32)
    masked_loss = pair_loss * valid_pairs

    # 平均损失
    n_pairs = tf.reduce_sum(valid_pairs) + 1e-8
    loss = tf.reduce_sum(masked_loss) / n_pairs

    return loss


def listnet_loss(selection_scores, cut_qualities, temperature=1.0):
    """
    ListNet损失 - 学习排序的state-of-the-art方法

    核心思想：
    1. 将选择分数通过softmax转换为概率分布
    2. 将真实质量也通过softmax转换为概率分布
    3. 最小化两个分布的交叉熵

    优势：
    - 比pairwise hinge loss更强，直接优化整体排序
    - 对噪声更鲁棒
    - 梯度更稳定

    Args:
        selection_scores: [N_cuts] 所有割平面的选择分数
        cut_qualities: [N_cuts] 每个割平面的质量
        temperature: float 温度参数，控制分布的平滑度

    Returns:
        loss: scalar
    """
    # 将选择分数转换为Top-1概率分布
    pred_probs = tf.nn.softmax(selection_scores / temperature)

    # 将真实质量转换为理想概率分布
    true_probs = tf.nn.softmax(cut_qualities / temperature)

    # 交叉熵损失: -sum(true_probs * log(pred_probs))
    epsilon = 1e-8
    cross_entropy = -tf.reduce_sum(true_probs * tf.math.log(pred_probs + epsilon))

    return cross_entropy


def combined_ranking_loss(selection_scores, cut_qualities,
                         use_listnet=True, use_pairwise=True,
                         lambda_listnet=1.0, lambda_pairwise=0.5):
    """
    组合排序损失：ListNet + Pairwise Hinge

    结合两种损失的优势：
    - ListNet: 全局排序优化
    - Pairwise: 保证局部正确性

    Args:
        selection_scores: [N_cuts] 选择分数
        cut_qualities: [N_cuts] 真实质量
        use_listnet: bool 是否使用ListNet损失
        use_pairwise: bool 是否使用Pairwise损失
        lambda_listnet: float ListNet权重
        lambda_pairwise: float Pairwise权重

    Returns:
        loss: scalar
    """
    total_loss = 0.0

    if use_listnet:
        L_listnet = listnet_loss(selection_scores, cut_qualities)
        total_loss += lambda_listnet * L_listnet

    if use_pairwise:
        L_pairwise = selection_ranking_loss(selection_scores, cut_qualities)
        total_loss += lambda_pairwise * L_pairwise

    return total_loss



def compute_cut_qualities(subsets_data, N_cuts):
    """
    计算每个割平面的质量

    策略：在高质量子集中频繁出现的割平面质量高

    Args:
        subsets_data: List of dicts, 每个dict包含:
            - 'indices': List[int] or tensor 子集索引
            - 'joint_improvement': float 联合下界提升
        N_cuts: int 割平面总数

    Returns:
        qualities: [N_cuts] tensor, 每个割平面的质量分数
    """
    import numpy as np

    # 按下界提升排序
    sorted_subsets = sorted(subsets_data, key=lambda s: s['joint_improvement'], reverse=True)

    # 计算每个割平面的加权出现次数
    qualities = np.zeros(N_cuts, dtype=np.float32)

    for rank, subset in enumerate(sorted_subsets):
        # 权重：排名越靠前，权重越大
        weight = 1.0 / (rank + 1)

        indices = subset['indices']
        # 处理tensor和list两种情况
        if isinstance(indices, tf.Tensor):
            indices = indices.numpy()
        if isinstance(indices, np.ndarray):
            indices = indices.tolist()

        for cut_idx in indices:
            qualities[cut_idx] += weight

    # 归一化
    qualities = qualities / (qualities.max() + 1e-8)

    return tf.constant(qualities, dtype=tf.float32)


def set_regression_loss(predictions, true_improvements):
    """
    集合值回归损失 (log-scale MSE)
    """
    epsilon = 1e-8
    log_pred = tf.math.log(predictions + epsilon)
    log_true = tf.math.log(true_improvements + epsilon)

    return tf.reduce_mean(tf.square(log_pred - log_true))


def set_contrastive_loss(predictions, true_improvements, margin=0.01):
    """
    集合对比损失 (与MILP_polished相同)
    """
    M = tf.shape(predictions)[0]

    pred_i = tf.expand_dims(predictions, 1)
    pred_j = tf.expand_dims(predictions, 0)
    pred_diff = pred_i - pred_j

    true_i = tf.expand_dims(true_improvements, 1)
    true_j = tf.expand_dims(true_improvements, 0)
    true_diff = true_i - true_j

    sign = tf.sign(true_diff)
    pair_loss = tf.maximum(0.0, -sign * pred_diff + margin)

    valid_pairs = tf.cast(tf.abs(true_diff) > 1e-8, tf.float32)
    masked_loss = pair_loss * valid_pairs

    n_pairs = tf.reduce_sum(valid_pairs) + 1e-8
    return tf.reduce_sum(masked_loss) / n_pairs


def combined_loss(model, state, subsets_data,
                 lambda_value=1.0, lambda_selection=2.0,
                 use_listnet=True):
    """
    组合损失

    L = lambda_value * L_value + lambda_selection * L_selection

    Args:
        model: DirectTopKSelector
        state: dict 图状态
        subsets_data: List of dicts 子集数据
        lambda_value: float 值预测损失权重
        lambda_selection: float 选择损失权重
        use_listnet: bool 是否使用ListNet损失（默认True）

    Returns:
        total_loss: scalar
        loss_dict: dict 各部分损失
    """
    N_cuts = int(state['cut_features'].shape[0])

    # 1. 值预测损失 (对所有子集)
    subsets_indices = [
        tf.constant(s['indices'], dtype=tf.int32)
        for s in subsets_data
    ]
    true_improvements = tf.constant(
        [s['joint_improvement'] for s in subsets_data],
        dtype=tf.float32
    )

    predictions = model.predict_for_subset(state, subsets_indices, training=True)

    L_regression = set_regression_loss(predictions, true_improvements)
    L_contrast = set_contrastive_loss(predictions, true_improvements)
    L_value = L_regression + L_contrast

    # 2. 选择损失
    # 计算每个割平面的质量
    cut_qualities = compute_cut_qualities(subsets_data, N_cuts)

    # 获取选择分数
    selection_scores = model(state, K=None, training=True)

    # 使用组合排序损失（ListNet + Pairwise）
    if use_listnet:
        L_selection = combined_ranking_loss(
            selection_scores, cut_qualities,
            use_listnet=True, use_pairwise=True,
            lambda_listnet=1.0, lambda_pairwise=0.5
        )
    else:
        # 仅使用Pairwise损失（向后兼容）
        L_selection = selection_ranking_loss(selection_scores, cut_qualities)

    # 3. 总损失
    total_loss = lambda_value * L_value + lambda_selection * L_selection

    loss_dict = {
        'total': total_loss,
        'value': L_value,
        'regression': L_regression,
        'contrastive': L_contrast,
        'selection': L_selection
    }

    return total_loss, loss_dict


# ========== 评估指标 ==========

def spearman_correlation(predictions, true_improvements):
    """Spearman相关系数"""
    pred_rank = tf.argsort(tf.argsort(predictions, direction='DESCENDING'))
    true_rank = tf.argsort(tf.argsort(true_improvements, direction='DESCENDING'))

    pred_rank = tf.cast(pred_rank, tf.float32)
    true_rank = tf.cast(true_rank, tf.float32)

    pred_centered = pred_rank - tf.reduce_mean(pred_rank)
    true_centered = true_rank - tf.reduce_mean(true_rank)

    numerator = tf.reduce_sum(pred_centered * true_centered)
    denominator = tf.sqrt(
        tf.reduce_sum(tf.square(pred_centered)) *
        tf.reduce_sum(tf.square(true_centered))
    )

    return numerator / (denominator + 1e-8)


def relative_error(predictions, true_improvements):
    """相对误差"""
    errors = tf.abs(predictions - true_improvements) / tf.maximum(true_improvements, 1e-8)
    return tf.reduce_mean(errors)


if __name__ == '__main__':
    import numpy as np

    # 测试损失函数
    N = 100
    M = 20

    # 模拟数据
    subsets_data = []
    for _ in range(M):
        size = np.random.choice([5, 10, 20])
        indices = np.random.choice(N, size=size, replace=False).tolist()
        improvement = np.random.uniform(0.001, 0.1)
        subsets_data.append({
            'indices': indices,
            'size': size,
            'joint_improvement': improvement
        })

    # 计算割平面质量
    qualities = compute_cut_qualities(subsets_data, N)
    print(f"[TEST] Cut qualities shape: {qualities.shape}")
    print(f"[TEST] Quality range: [{qualities.numpy().min():.3f}, {qualities.numpy().max():.3f}]")

    # 测试选择损失
    selection_scores = tf.constant(np.random.randn(N), dtype=tf.float32)
    L_sel = selection_ranking_loss(selection_scores, qualities)
    print(f"[TEST] Selection loss: {L_sel.numpy():.4f}")

    print("\n[TEST] All tests passed!")
