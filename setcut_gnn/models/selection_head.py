"""
选择头 - 计算每个割平面的选择分数
"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense


class SelectionHead(Model):
    """
    选择头：为每个割平面计算选择分数

    目标：分数高的割平面更有可能被选入Top-K
    """

    def __init__(self, emb_size=32):
        super().__init__()
        self.mlp = tf.keras.Sequential([
            Dense(emb_size, activation='relu'),
            Dense(emb_size // 2, activation='relu'),
            Dense(1)  # 输出选择分数（未归一化）
        ])

    def call(self, cut_embeddings, *, training=False):
        """
        Args:
            cut_embeddings: [N_cuts, emb_size]

        Returns:
            selection_scores: [N_cuts] 选择分数（logits）
        """
        scores = self.mlp(cut_embeddings, training=training)  # [N_cuts, 1]
        return tf.squeeze(scores, axis=-1)  # [N_cuts]


class AttentionPooling(Model):
    """
    注意力池化：聚合Top-K割平面的嵌入

    用于预测集合的联合下界提升
    """

    def __init__(self, emb_size=32):
        super().__init__()
        self.attention_mlp = Dense(1)
        self.emb_size = emb_size

    def call(self, embeddings, *, training=False):
        """
        Args:
            embeddings: [K, emb_size] Top-K割平面的嵌入

        Returns:
            pooled: [emb_size] 池化后的集合表示
        """
        # 计算注意力权重
        attention_logits = self.attention_mlp(embeddings, training=training)  # [K, 1]
        attention_weights = tf.nn.softmax(attention_logits, axis=0)  # [K, 1]

        # 加权求和
        pooled = tf.reduce_sum(embeddings * attention_weights, axis=0)  # [emb_size]

        return pooled


class SetValuePredictor(Model):
    """
    集合值预测器：预测Top-K割平面的联合下界提升
    """

    def __init__(self, emb_size=32):
        super().__init__()
        self.mlp = tf.keras.Sequential([
            Dense(emb_size, activation='relu'),
            Dense(emb_size // 2, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        self.output_scale = 0.1  # 缩放到合理范围

    def call(self, set_representation, *, training=False):
        """
        Args:
            set_representation: [emb_size] 集合的聚合表示

        Returns:
            joint_improvement: scalar 预测的联合下界提升
        """
        # 处理1D和2D输入
        if len(set_representation.shape) == 1:
            set_representation = tf.expand_dims(set_representation, 0)

        value = self.mlp(set_representation, training=training)  # [1, 1]
        value = value * self.output_scale

        # 返回标量
        return tf.squeeze(value)
