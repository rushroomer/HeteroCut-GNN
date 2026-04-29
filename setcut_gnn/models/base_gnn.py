"""
基础GNN编码器 - 从MILP_polished复制并简化
"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, LayerNormalization


class MessagePassingLayer(Model):
    """消息传递层"""

    def __init__(self, emb_size=32):
        super().__init__()
        self.emb_size = emb_size

        # 消息生成
        self.message_mlp = Dense(emb_size, activation='relu')

        # 门控更新
        self.gate_mlp = Dense(emb_size, activation='sigmoid')
        self.update_mlp = Dense(emb_size, activation='relu')

        # 归一化
        self.layer_norm = LayerNormalization()

    def call(self, node_feats, neighbor_feats, edge_indices, edge_weights, *, reverse=False, training=False):
        """
        Args:
            node_feats: [N_dst, emb_size]
            neighbor_feats: [N_src, emb_size]
            edge_indices: [2, E] (src_idx, dst_idx)
            edge_weights: [E, 1]
            reverse: bool (keyword-only)
            training: bool (keyword-only)

        Returns:
            updated_feats: [N_dst, emb_size]
        """
        # 从node_feats推断节点数量
        n_nodes = tf.shape(node_feats)[0]

        # 提取边的索引
        if not reverse:
            src_idx = edge_indices[0]
            dst_idx = edge_indices[1]
        else:
            src_idx = edge_indices[1]
            dst_idx = edge_indices[0]

        # 生成消息
        neighbor_messages = self.message_mlp(neighbor_feats, training=training)  # [N_src, emb_size]
        messages = tf.gather(neighbor_messages, src_idx)  # [E, emb_size]

        # 边权重加权
        weighted_messages = messages * edge_weights  # [E, emb_size]

        # 聚合消息
        aggregated = tf.scatter_nd(
            indices=tf.expand_dims(dst_idx, 1),
            updates=weighted_messages,
            shape=[n_nodes, self.emb_size]
        )

        # 归一化（按度数）
        degree = tf.scatter_nd(
            indices=tf.expand_dims(dst_idx, 1),
            updates=edge_weights,
            shape=[n_nodes, 1]
        )
        aggregated = aggregated / tf.maximum(degree, 1.0)

        # 门控更新
        gate = self.gate_mlp(tf.concat([aggregated, node_feats], axis=-1), training=training)
        update = self.update_mlp(tf.concat([aggregated, node_feats], axis=-1), training=training)

        # 残差连接
        output = gate * update + (1 - gate) * node_feats

        # 层归一化
        return self.layer_norm(output)


class EdgeWeightModule(Model):
    """边权重学习模块"""

    def __init__(self, emb_size=32):
        super().__init__()
        self.mlp = Dense(1, activation='sigmoid')

    def call(self, src_feats, dst_feats, edge_feats):
        """
        Args:
            src_feats: [E, emb_size]
            dst_feats: [E, emb_size]
            edge_feats: [E, feat_dim]

        Returns:
            weights: [E, 1]
        """
        combined = tf.concat([src_feats, dst_feats, edge_feats], axis=-1)
        return self.mlp(combined, training=False)  # Edge weights不需要dropout


class BaseGNNEncoder(Model):
    """基础GNN编码器"""

    def __init__(self, emb_size=32, n_layers=3):
        super().__init__()
        self.emb_size = emb_size
        self.n_layers = n_layers

        # 节点嵌入
        self.var_embedding = Dense(emb_size, activation='relu', name='var_embedding')
        self.cons_embedding = Dense(emb_size, activation='relu', name='cons_embedding')
        self.cut_embedding = Dense(emb_size, activation='relu', name='cut_embedding')

        # 边权重模块
        self.edge_weight_vc = EdgeWeightModule(emb_size)
        self.edge_weight_vk = EdgeWeightModule(emb_size)

        # 消息传递层
        self.mp_v2c = [MessagePassingLayer(emb_size) for _ in range(n_layers)]
        self.mp_c2v = [MessagePassingLayer(emb_size) for _ in range(n_layers)]
        self.mp_v2k = [MessagePassingLayer(emb_size) for _ in range(n_layers)]

    def call(self, state, *, training=False):
        """
        Args:
            state: dict {
                'variable_features': [N_var, var_feat_dim],
                'constraint_features': [N_cons, cons_feat_dim],
                'cut_features': [N_cuts, cut_feat_dim],
                'var_cons_edges': [2, E_vc],
                'var_cons_edge_features': [E_vc, edge_feat_dim],
                'var_cut_edges': [2, E_vk],
                'var_cut_edge_features': [E_vk, edge_feat_dim]
            }
            training: bool (keyword-only)

        Returns:
            cut_embeddings: [N_cuts, emb_size]
        """
        # 初始嵌入
        h_var = self.var_embedding(state['variable_features'], training=training)
        h_cons = self.cons_embedding(state['constraint_features'], training=training)
        h_cut = self.cut_embedding(state['cut_features'], training=training)

        # 计算边权重
        vc_edges = state['var_cons_edges']
        vk_edges = state['var_cut_edges']

        # var-cons 边权重
        vc_src_feats = tf.gather(h_var, vc_edges[0])
        vc_dst_feats = tf.gather(h_cons, vc_edges[1])
        vc_weights = self.edge_weight_vc(
            vc_src_feats, vc_dst_feats, state['var_cons_edge_features']
        )

        # var-cut 边权重
        vk_src_feats = tf.gather(h_var, vk_edges[0])
        vk_dst_feats = tf.gather(h_cut, vk_edges[1])
        vk_weights = self.edge_weight_vk(
            vk_src_feats, vk_dst_feats, state['var_cut_edge_features']
        )

        # 消息传递
        for layer in range(self.n_layers):
            # var → cons
            h_cons = self.mp_v2c[layer](h_cons, h_var, vc_edges, vc_weights, reverse=False, training=training)

            # cons → var
            h_var = self.mp_c2v[layer](h_var, h_cons, vc_edges, vc_weights, reverse=True, training=training)

            # var → cut
            h_cut = self.mp_v2k[layer](h_cut, h_var, vk_edges, vk_weights, reverse=False, training=training)

        return h_cut  # [N_cuts, emb_size]
