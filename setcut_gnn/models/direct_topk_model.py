"""
直接Top-K割平面选择模型 - 核心创新

与MILP_polished的关键区别：
1. 不使用贪心搜索（避免次模假设）
2. 直接输出Top-K个割平面
3. 同时预测联合下界提升
4. 一次前向传播完成选择
5. 支持30维割平面特征和双向消息传递
"""

import pickle
import tensorflow as tf
from tensorflow.keras import Model

# 使用增强版GNN（支持双向消息传递）
try:
    from models.enhanced_gnn import EnhancedGNNEncoder as BaseGNNEncoder
except ImportError:
    from models.base_gnn import BaseGNNEncoder

from models.selection_head import SelectionHead, AttentionPooling, SetValuePredictor


class DirectTopKSelector(Model):
    """
    直接Top-K选择器

    输入：
        - state: 异构图（包含N个候选割平面）
        - K: 目标选择数量

    输出：
        - top_k_indices: [K] 选中的割平面索引
        - selection_scores: [N] 每个割平面的选择分数
        - joint_improvement: scalar 预测的K个割平面的联合下界提升
    """

    def __init__(self, emb_size=32, n_layers=3):
        super().__init__()
        self.emb_size = emb_size
        self.n_layers = n_layers

        # 基础GNN编码器
        self.base_gnn = BaseGNNEncoder(emb_size=emb_size, n_layers=n_layers)

        # 选择头
        self.selection_head = SelectionHead(emb_size=emb_size)

        # 集合表示 & 值预测
        self.attention_pooling = AttentionPooling(emb_size=emb_size)
        self.set_value_predictor = SetValuePredictor(emb_size=emb_size)

    def call(self, state, K=None, *, training=False):
        """
        Args:
            state: dict 图状态
            K: int 或 None
                - 如果提供：执行Top-K选择并预测联合下界提升
                - 如果为None：只计算选择分数（训练时使用）

        Returns:
            如果K is not None:
                (top_k_indices, selection_scores, joint_improvement)
            如果K is None:
                selection_scores
        """
        # 1. GNN编码所有割平面
        cut_embeddings = self.base_gnn(state, training=training)  # [N_cuts, emb_size]

        # 2. 计算选择分数
        selection_scores = self.selection_head(cut_embeddings, training=training)  # [N_cuts]

        # 3. 如果需要Top-K选择
        if K is not None:
            # Top-K选择
            top_k_values, top_k_indices = tf.nn.top_k(selection_scores, k=K)

            # 提取Top-K的嵌入
            top_k_embeddings = tf.gather(cut_embeddings, top_k_indices)  # [K, emb_size]

            # 聚合表示
            set_representation = self.attention_pooling(top_k_embeddings, training=training)

            # 预测联合下界提升
            joint_improvement = self.set_value_predictor(set_representation, training=training)

            return top_k_indices, selection_scores, joint_improvement

        # 训练时只返回分数
        return selection_scores

    def predict_for_subset(self, state, subset_indices, training=False):
        """
        为给定子集预测联合下界提升（训练时使用）

        Args:
            state: dict 图状态
            subset_indices: [M] 或 List of [K_i] 子集索引

        Returns:
            predictions: [M] 或 [num_subsets] 预测的联合下界提升
        """
        # 1. GNN编码
        cut_embeddings = self.base_gnn(state, training=training)  # [N_cuts, emb_size]

        # 2. 处理单个子集或多个子集
        if isinstance(subset_indices, list):
            # 多个子集
            predictions = []
            for indices in subset_indices:
                # 提取子集嵌入
                subset_embs = tf.gather(cut_embeddings, indices)  # [K_i, emb_size]

                # 聚合 & 预测
                set_repr = self.attention_pooling(subset_embs, training=training)
                pred = self.set_value_predictor(set_repr, training=training)
                predictions.append(pred)

            return tf.stack(predictions)  # [num_subsets]
        else:
            # 单个子集
            subset_embs = tf.gather(cut_embeddings, subset_indices)
            set_repr = self.attention_pooling(subset_embs, training=training)
            return self.set_value_predictor(set_repr, training=training)

    def save_model(self, filepath):
        """保存模型权重"""
        weights = self.get_weights()
        with open(filepath, 'wb') as f:
            pickle.dump(weights, f)

    def load_model(self, filepath):
        """加载模型权重"""
        with open(filepath, 'rb') as f:
            weights = pickle.load(f)
        self.set_weights(weights)


if __name__ == '__main__':
    # 测试模型
    import numpy as np

    # 模拟状态
    N_vars = 100
    N_cons = 50
    N_cuts = 80
    E_vc = 300
    E_vk = 200

    # 生成随机边
    vc_src = np.random.randint(0, N_vars, size=E_vc)
    vc_dst = np.random.randint(0, N_cons, size=E_vc)
    vk_src = np.random.randint(0, N_vars, size=E_vk)
    vk_dst = np.random.randint(0, N_cuts, size=E_vk)

    state = {
        'variable_features': tf.constant(np.random.randn(N_vars, 14), dtype=tf.float32),
        'constraint_features': tf.constant(np.random.randn(N_cons, 3), dtype=tf.float32),
        'cut_features': tf.constant(np.random.randn(N_cuts, 3), dtype=tf.float32),
        'var_cons_edges': tf.constant(np.stack([vc_src, vc_dst], axis=0), dtype=tf.int32),
        'var_cons_edge_features': tf.constant(np.random.randn(E_vc, 1), dtype=tf.float32),
        'var_cut_edges': tf.constant(np.stack([vk_src, vk_dst], axis=0), dtype=tf.int32),
        'var_cut_edge_features': tf.constant(np.random.randn(E_vk, 1), dtype=tf.float32)
    }

    # 创建模型
    model = DirectTopKSelector(emb_size=32, n_layers=3)

    # 测试1：Top-K选择
    print("=" * 60)
    print("测试1：Top-K选择（推理模式）")
    print("=" * 60)
    top_k_indices, selection_scores, joint_improvement = model(state, K=10, training=False)
    print(f"选中的割平面索引: {top_k_indices.numpy()}")
    print(f"选择分数形状: {selection_scores.shape}")
    print(f"预测联合下界提升: {joint_improvement.numpy():.6f}")

    # 测试2：训练模式（预测子集）
    print("\n" + "=" * 60)
    print("测试2：子集预测（训练模式）")
    print("=" * 60)
    subset_indices = [
        tf.constant([0, 5, 10, 15, 20], dtype=tf.int32),
        tf.constant([1, 6, 11, 16, 21, 26, 31, 36, 41, 46], dtype=tf.int32)
    ]
    predictions = model.predict_for_subset(state, subset_indices, training=True)
    print(f"预测值: {predictions.numpy()}")

    print("\n[TEST] All tests passed!")
