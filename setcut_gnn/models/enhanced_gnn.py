"""
增强版GNN编码器 - 支持双向消息传递

关键改进：
1. 添加 cut → var 反向传播
2. 支持30维割平面特征
3. 更深的消息传递层
"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, LayerNormalization, Dropout


class MessagePassingLayer(Model):
    """消息传递层（带门控和残差）"""

    def __init__(self, emb_size=64, dropout_rate=0.1):
        super().__init__()
        self.emb_size = emb_size

        # 消息生成
        self.message_mlp = tf.keras.Sequential([
            Dense(emb_size, activation='relu'),
            Dropout(dropout_rate),
            Dense(emb_size)
        ])

        # 门控更新
        self.gate_mlp = Dense(emb_size, activation='sigmoid')
        self.update_mlp = tf.keras.Sequential([
            Dense(emb_size, activation='relu'),
            Dropout(dropout_rate),
            Dense(emb_size)
        ])

        # 归一化
        self.layer_norm = LayerNormalization()

    def call(self, node_feats, neighbor_feats, edge_indices, edge_weights, *, reverse=False, training=False):
        """
        Args:
            node_feats: [N_dst, emb_size]
            neighbor_feats: [N_src, emb_size]
            edge_indices: [2, E] (src_idx, dst_idx)
            edge_weights: [E, 1]
            reverse: bool 是否反向传播
            training: bool

        Returns:
            updated_feats: [N_dst, emb_size]
        """
        n_nodes = tf.shape(node_feats)[0]

        if not reverse:
            src_idx = edge_indices[0]
            dst_idx = edge_indices[1]
        else:
            src_idx = edge_indices[1]
            dst_idx = edge_indices[0]

        # 生成消息
        neighbor_messages = self.message_mlp(neighbor_feats, training=training)
        messages = tf.gather(neighbor_messages, src_idx)

        # 边权重加权
        weighted_messages = messages * edge_weights

        # 聚合消息
        aggregated = tf.scatter_nd(
            indices=tf.expand_dims(dst_idx, 1),
            updates=weighted_messages,
            shape=[n_nodes, self.emb_size]
        )

        # 按度数归一化
        degree = tf.scatter_nd(
            indices=tf.expand_dims(dst_idx, 1),
            updates=edge_weights,
            shape=[n_nodes, 1]
        )
        aggregated = aggregated / tf.maximum(degree, 1.0)

        # 门控更新
        combined = tf.concat([aggregated, node_feats], axis=-1)
        gate = self.gate_mlp(combined, training=training)
        update = self.update_mlp(combined, training=training)

        # 残差连接
        output = gate * update + (1 - gate) * node_feats

        return self.layer_norm(output)


class EdgeWeightModule(Model):
    """边权重学习模块"""

    def __init__(self, emb_size=64):
        super().__init__()
        self.mlp = tf.keras.Sequential([
            Dense(emb_size, activation='relu'),
            Dense(1, activation='sigmoid')
        ])

    def call(self, src_feats, dst_feats, edge_feats):
        combined = tf.concat([src_feats, dst_feats, edge_feats], axis=-1)
        return self.mlp(combined, training=False)


class EnhancedGNNEncoder(Model):
    """
    增强版GNN编码器

    消息传递路径：
    1. var → cons → var（双向）
    2. var → cut → var（双向，新增cut→var）
    """

    def __init__(self, emb_size=64, n_layers=4, dropout_rate=0.1):
        super().__init__()
        self.emb_size = emb_size
        self.n_layers = n_layers

        # 节点嵌入（支持不同维度的输入）
        self.var_embedding = tf.keras.Sequential([
            Dense(emb_size, activation='relu'),
            Dense(emb_size)
        ], name='var_embedding')

        self.cons_embedding = tf.keras.Sequential([
            Dense(emb_size, activation='relu'),
            Dense(emb_size)
        ], name='cons_embedding')

        self.cut_embedding = tf.keras.Sequential([
            Dense(emb_size, activation='relu'),
            Dense(emb_size)
        ], name='cut_embedding')

        # 边权重模块
        self.edge_weight_vc = EdgeWeightModule(emb_size)
        self.edge_weight_vk = EdgeWeightModule(emb_size)

        # 消息传递层
        self.mp_v2c = [MessagePassingLayer(emb_size, dropout_rate) for _ in range(n_layers)]
        self.mp_c2v = [MessagePassingLayer(emb_size, dropout_rate) for _ in range(n_layers)]
        self.mp_v2k = [MessagePassingLayer(emb_size, dropout_rate) for _ in range(n_layers)]
        self.mp_k2v = [MessagePassingLayer(emb_size, dropout_rate) for _ in range(n_layers)]  # 新增

    def call(self, state, *, training=False):
        """
        Args:
            state: dict {
                'variable_features': [N_var, var_feat_dim],
                'constraint_features': [N_cons, cons_feat_dim],
                'cut_features': [N_cuts, cut_feat_dim],  # 支持3维或30维
                'var_cons_edges': [2, E_vc],
                'var_cons_edge_features': [E_vc, edge_feat_dim],
                'var_cut_edges': [2, E_vk],
                'var_cut_edge_features': [E_vk, edge_feat_dim]
            }
            training: bool

        Returns:
            cut_embeddings: [N_cuts, emb_size]
        """
        # 初始嵌入
        h_var = self.var_embedding(state['variable_features'], training=training)
        h_cons = self.cons_embedding(state['constraint_features'], training=training)
        h_cut = self.cut_embedding(state['cut_features'], training=training)

        # 边信息
        vc_edges = state['var_cons_edges']
        vk_edges = state['var_cut_edges']

        # 计算边权重
        vc_src_feats = tf.gather(h_var, vc_edges[0])
        vc_dst_feats = tf.gather(h_cons, vc_edges[1])
        vc_weights = self.edge_weight_vc(
            vc_src_feats, vc_dst_feats, state['var_cons_edge_features']
        )

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

            # cut → var (新增：反向传播，让变量感知割平面信息)
            h_var = self.mp_k2v[layer](h_var, h_cut, vk_edges, vk_weights, reverse=True, training=training)

        return h_cut


# 兼容旧版本
BaseGNNEncoder = EnhancedGNNEncoder


if __name__ == '__main__':
    import numpy as np

    # 测试
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

    # 测试30维割平面特征
    state = {
        'variable_features': tf.constant(np.random.randn(N_vars, 14), dtype=tf.float32),
        'constraint_features': tf.constant(np.random.randn(N_cons, 3), dtype=tf.float32),
        'cut_features': tf.constant(np.random.randn(N_cuts, 30), dtype=tf.float32),  # 30维
        'var_cons_edges': tf.constant(np.stack([vc_src, vc_dst], axis=0), dtype=tf.int32),
        'var_cons_edge_features': tf.constant(np.random.randn(E_vc, 1), dtype=tf.float32),
        'var_cut_edges': tf.constant(np.stack([vk_src, vk_dst], axis=0), dtype=tf.int32),
        'var_cut_edge_features': tf.constant(np.random.randn(E_vk, 1), dtype=tf.float32)
    }

    # 创建模型
    encoder = EnhancedGNNEncoder(emb_size=64, n_layers=4)
    cut_emb = encoder(state, training=False)

    print(f"输入割平面特征维度: {state['cut_features'].shape}")
    print(f"输出割平面嵌入维度: {cut_emb.shape}")
    print("[TEST] EnhancedGNNEncoder测试通过！")
