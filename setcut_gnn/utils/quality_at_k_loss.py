"""
Quality@K损失函数

直接优化模型选择Top-K的质量，而非单个排序

核心思想：
1. 让模型选择Top-K
2. 用值预测head近似模型Top-K的联合下界提升
3. 与Efficacy Top-K的下界提升对比
4. 最大化Quality@K = model_improvement / efficacy_improvement
"""

import tensorflow as tf
import numpy as np


def compute_efficacy_topk(subsets_data, N_cuts, K):
    """
    计算Efficacy Top-K

    Args:
        subsets_data: List[dict] 子集数据
        N_cuts: int 割平面总数
        K: int 选择数量

    Returns:
        efficacy_topk_indices: tf.Tensor [K] Efficacy最高的K个割平面索引
        efficacy_improvement: float Efficacy Top-K的下界提升
    """
    # 计算每个割平面的efficacy（单个质量）
    cut_qualities = np.zeros(N_cuts, dtype=np.float32)
    cut_counts = np.zeros(N_cuts, dtype=np.float32)

    for subset in subsets_data:
        if subset['size'] == 1:  # 单个割平面的下界提升 = efficacy
            idx = subset['indices'][0]
            cut_qualities[idx] = subset['joint_improvement']
            cut_counts[idx] += 1

    # 如果没有单个割平面的数据，用子集平均估计
    for i in range(N_cuts):
        if cut_counts[i] == 0:
            # 找包含该割平面的所有子集，取平均
            containing_subsets = [
                s for s in subsets_data
                if i in s['indices']
            ]
            if containing_subsets:
                cut_qualities[i] = np.mean([
                    s['joint_improvement'] / s['size']
                    for s in containing_subsets
                ])

    # Efficacy Top-K
    efficacy_topk_indices = np.argsort(cut_qualities)[-K:]

    # 找subsets_data中最接近efficacy_topk的子集（size=K）
    k_subsets = [s for s in subsets_data if s['size'] == K]

    if k_subsets:
        # 找与efficacy_topk重叠最多的子集
        efficacy_set = set(efficacy_topk_indices)
        best_match = max(
            k_subsets,
            key=lambda s: len(set(s['indices']) & efficacy_set)
        )
        efficacy_improvement = best_match['joint_improvement']
    else:
        # 如果没有size=K的子集，用单个质量之和估计（会低估）
        efficacy_improvement = np.sum(cut_qualities[efficacy_topk_indices])

    return tf.constant(efficacy_topk_indices, dtype=tf.int32), efficacy_improvement


def quality_at_k_loss(model, state, subsets_data, K=5, target_ratio=2.0, loss_type='margin', training=True):
    """
    Quality@K损失：直接优化模型Top-K vs Efficacy Top-K

    Args:
        model: DirectTopKSelector 模型
        state: dict 图状态
        subsets_data: List[dict] 子集数据
        K: int 选择数量
        target_ratio: float 目标Quality@K（默认2.0 = 超过Efficacy 100%）
        loss_type: str 损失类型
            - 'margin': Margin Loss，鼓励model_improvement ≥ target_ratio * efficacy
            - 'ratio': Ratio Loss，直接最大化 model/efficacy
            - 'huber': Smooth L1 Loss
        training: bool 是否训练模式

    Returns:
        loss: scalar
        info: dict 损失信息
    """
    N_cuts = int(state['cut_features'].shape[0])

    # 检查：如果割平面数量不足K，返回零损失
    if N_cuts < K:
        return tf.constant(0.0, dtype=tf.float32), {
            'quality_k': 0.0,
            'model_improvement': 0.0,
            'efficacy_improvement': 0.0,
            'overlap': 0.0,
            'loss_value': 0.0,
            'skipped': True,
            'reason': f'N_cuts={N_cuts} < K={K}'
        }

    # 1. 模型选择Top-K
    selection_scores = model(state, K=None, training=training)  # [N_cuts]
    model_topk_indices = tf.math.top_k(selection_scores, k=K).indices  # [K]

    # 2. 用值预测head近似模型Top-K的联合下界提升
    model_topk_indices_list = [model_topk_indices]
    model_improvement_pred = model.predict_for_subset(
        state, model_topk_indices_list, training=training
    )[0]  # scalar

    # 3. 计算Efficacy Top-K
    efficacy_topk_indices, efficacy_improvement = compute_efficacy_topk(
        subsets_data, N_cuts, K
    )
    efficacy_improvement = tf.constant(efficacy_improvement, dtype=tf.float32)

    # 4. 计算损失
    if loss_type == 'margin':
        # Margin Loss: max(0, target * efficacy - model)
        # 目标：model ≥ target_ratio * efficacy
        target_improvement = target_ratio * efficacy_improvement
        loss = tf.maximum(0.0, target_improvement - model_improvement_pred)

        # 归一化（避免不同样本scale差异太大）
        loss = loss / (efficacy_improvement + 1e-8)

    elif loss_type == 'ratio':
        # Ratio Loss: -log(model / efficacy)
        # 目标：最大化 model / efficacy
        quality_k = model_improvement_pred / (efficacy_improvement + 1e-8)
        loss = -tf.math.log(quality_k + 1e-8)

    elif loss_type == 'huber':
        # Huber Loss: smooth L1 between model and target
        target_improvement = target_ratio * efficacy_improvement
        loss = tf.keras.losses.huber(
            target_improvement,
            model_improvement_pred,
            delta=0.1 * efficacy_improvement  # delta相对于efficacy
        )

        # 归一化
        loss = loss / (efficacy_improvement + 1e-8)

    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    # 5. 计算实际Quality@K（用于监控）
    quality_k = model_improvement_pred / (efficacy_improvement + 1e-8)

    # 6. 计算Top-K重叠率（用于监控）
    model_set = set(model_topk_indices.numpy())
    efficacy_set = set(efficacy_topk_indices.numpy())
    overlap = len(model_set & efficacy_set) / K

    info = {
        'quality_k': quality_k.numpy(),
        'model_improvement': model_improvement_pred.numpy(),
        'efficacy_improvement': efficacy_improvement.numpy(),
        'overlap': overlap,
        'loss_value': loss.numpy()
    }

    return loss, info


def multi_k_quality_loss(model, state, subsets_data,
                         K_values=[5, 10, 20],
                         K_weights=[2.0, 1.0, 0.5],
                         target_ratio=2.0,
                         loss_type='margin',
                         training=True):
    """
    多个K值的Quality@K联合损失

    Args:
        K_values: List[int] 要优化的K值列表
        K_weights: List[float] 每个K的权重（K=5更重要）
        training: bool 是否训练模式
        其他参数同quality_at_k_loss

    Returns:
        total_loss: scalar
        info_dict: dict 各K的详细信息
    """
    total_loss = 0.0
    info_dict = {}

    for K, weight in zip(K_values, K_weights):
        loss_k, info_k = quality_at_k_loss(
            model, state, subsets_data,
            K=K, target_ratio=target_ratio, loss_type=loss_type, training=training
        )
        total_loss += weight * loss_k
        info_dict[f'K={K}'] = info_k

    # 归一化
    total_loss = total_loss / sum(K_weights)

    return total_loss, info_dict


def combined_quality_and_value_loss(model, state, subsets_data,
                                    lambda_quality=2.0,
                                    lambda_value=1.0,
                                    K_values=[5, 10, 20],
                                    target_ratio=2.0,
                                    training=True):
    """
    组合Quality@K损失和Value损失

    思路：
    - Quality@K损失：直接优化Top-K选择
    - Value损失：辅助训练，帮助值预测head学习准确预测

    Args:
        lambda_quality: float Quality@K损失权重
        lambda_value: float Value损失权重
        training: bool 是否训练模式
        其他参数同上

    Returns:
        total_loss: scalar
        loss_dict: dict 各部分损失
    """
    # 1. Quality@K损失
    L_quality, quality_info = multi_k_quality_loss(
        model, state, subsets_data,
        K_values=K_values,
        target_ratio=target_ratio,
        loss_type='margin',
        training=training
    )

    # 2. Value损失（保留，帮助值预测head训练）
    from .losses import set_regression_loss, set_contrastive_loss

    subsets_indices = [
        tf.constant(s['indices'], dtype=tf.int32)
        for s in subsets_data
    ]
    true_improvements = tf.constant(
        [s['joint_improvement'] for s in subsets_data],
        dtype=tf.float32
    )

    predictions = model.predict_for_subset(state, subsets_indices, training=training)

    L_regression = set_regression_loss(predictions, true_improvements)
    L_contrast = set_contrastive_loss(predictions, true_improvements)
    L_value = L_regression + L_contrast

    # 3. 总损失
    total_loss = lambda_quality * L_quality + lambda_value * L_value

    loss_dict = {
        'total': total_loss.numpy(),
        'quality': L_quality.numpy(),
        'value': L_value.numpy(),
        'regression': L_regression.numpy(),
        'contrastive': L_contrast.numpy(),
        **quality_info
    }

    return total_loss, loss_dict
