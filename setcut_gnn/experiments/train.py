"""
DirectTopKSelector训练脚本

数据：复用MILP_polished的集合级数据
"""

import glob
import gzip
import os
import pickle
from datetime import timedelta
from time import perf_counter

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

import sys
sys.path.insert(0, '/home/jjw/milp_yxy/MILP_set_1222')

from models.direct_topk_model import DirectTopKSelector
from utils.losses import combined_loss, spearman_correlation, relative_error


def load_sample(filename):
    """加载样本"""
    with gzip.open(filename, 'rb') as f:
        return pickle.load(f)


def prepare_batch(sample_files):
    """准备训练批次"""
    for filename in sample_files:
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

        yield state_tensor, subsets_data


def train_step(model, optimizer, state, subsets_data, lambda_value, lambda_selection):
    """单步训练"""
    with tf.GradientTape() as tape:
        # 计算损失
        total_loss, loss_dict = combined_loss(
            model, state, subsets_data,
            lambda_value=lambda_value,
            lambda_selection=lambda_selection
        )

    # 反向传播
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return total_loss, loss_dict


def evaluate(model, sample_files, lambda_value=1.0, lambda_selection=2.0):
    """评估模型"""
    total_loss = 0.0
    total_corr = 0.0
    total_error = 0.0
    n_samples = 0

    for state, subsets_data in prepare_batch(sample_files):
        # 计算损失（使用相同的lambda参数）
        loss, loss_dict = combined_loss(
            model, state, subsets_data,
            lambda_value=lambda_value,
            lambda_selection=lambda_selection
        )

        # 计算指标（基于值预测）
        subsets_indices = [
            tf.constant(s['indices'], dtype=tf.int32)
            for s in subsets_data
        ]
        true_improvements = tf.constant(
            [s['joint_improvement'] for s in subsets_data],
            dtype=tf.float32
        )
        predictions = model.predict_for_subset(state, subsets_indices, training=False)

        corr = spearman_correlation(predictions, true_improvements)
        error = relative_error(predictions, true_improvements)

        total_loss += loss.numpy()
        total_corr += corr.numpy()
        total_error += error.numpy()
        n_samples += 1

    metrics = {
        'loss': total_loss / n_samples,
        'spearman': total_corr / n_samples,
        'relative_error': total_error / n_samples
    }

    return metrics


def train_model(train_files, valid_files, save_dir,
                emb_size=32, n_layers=3, max_epochs=100,
                lr=1e-4, lambda_value=1.0, lambda_selection=2.0,
                patience=10, early_stopping=20):
    """训练DirectTopKSelector模型"""
    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, 'train.log')

    def log(msg):
        print(msg)
        with open(log_file, 'a') as f:
            f.write(msg + '\n')

    log("=" * 80)
    log("DirectTopKSelector模型训练")
    log("=" * 80)
    log(f"训练样本数: {len(train_files)}")
    log(f"验证样本数: {len(valid_files)}")
    log(f"嵌入维度: {emb_size}")
    log(f"GNN 层数: {n_layers}")
    log(f"学习率: {lr}")
    log(f"值损失权重: {lambda_value}")
    log(f"选择损失权重: {lambda_selection}")
    log("")

    # 创建模型
    model = DirectTopKSelector(emb_size=emb_size, n_layers=n_layers)
    optimizer = Adam(learning_rate=lr)

    best_loss = np.inf
    plateau_count = 0
    start_time = perf_counter()

    for epoch in range(1, max_epochs + 1):
        log(f"Epoch {epoch}/{max_epochs}")

        # 训练
        train_metrics = {
            'loss': [],
            'value_loss': [],
            'selection_loss': []
        }

        # 随机打乱训练集
        np.random.shuffle(train_files)

        for i, (state, subsets_data) in enumerate(prepare_batch(train_files)):
            try:
                loss, loss_dict = train_step(
                    model, optimizer, state, subsets_data,
                    lambda_value, lambda_selection
                )

                train_metrics['loss'].append(loss.numpy())
                train_metrics['value_loss'].append(loss_dict['value'].numpy())
                train_metrics['selection_loss'].append(loss_dict['selection'].numpy())

                if (i + 1) % 50 == 0:
                    log(f"  [Train] Batch {i+1}/{len(train_files)} - "
                        f"Loss: {loss.numpy():.4f} "
                        f"(Value: {loss_dict['value'].numpy():.4f}, "
                        f"Selection: {loss_dict['selection'].numpy():.4f})")

            except Exception as e:
                log(f"  [WARNING] Batch {i} failed: {e}")
                continue

        # 平均训练指标
        avg_train_metrics = {
            k: np.mean(v) for k, v in train_metrics.items()
        }

        log(f"  [Train] Avg Loss: {avg_train_metrics['loss']:.4f} "
            f"(Value: {avg_train_metrics['value_loss']:.4f}, "
            f"Selection: {avg_train_metrics['selection_loss']:.4f})")

        # 验证
        valid_metrics = evaluate(model, valid_files, lambda_value, lambda_selection)
        log(f"  [Valid] Loss: {valid_metrics['loss']:.4f}, "
            f"Spearman: {valid_metrics['spearman']:.4f}, "
            f"Error: {valid_metrics['relative_error']:.4f}")

        # 保存最佳模型
        if valid_metrics['loss'] < best_loss:
            best_loss = valid_metrics['loss']
            plateau_count = 0
            model.save_model(os.path.join(save_dir, 'best_model.pkl'))
            log("  *** 最佳模型已保存 ***")
        else:
            plateau_count += 1
            if plateau_count >= early_stopping:
                log(f"  早停：{early_stopping} 轮无改进")
                break
            if plateau_count % patience == 0:
                lr *= 0.5
                optimizer.learning_rate.assign(lr)
                log(f"  学习率降至: {lr:.2e}")

        log("")

    # 加载最佳模型
    model.load_model(os.path.join(save_dir, 'best_model.pkl'))

    # 最终评估
    log("=" * 80)
    log("最终评估（最佳模型）")
    log("=" * 80)
    final_metrics = evaluate(model, valid_files, lambda_value, lambda_selection)
    log(f"Loss: {final_metrics['loss']:.4f}")
    log(f"Spearman: {final_metrics['spearman']:.4f}")
    log(f"Relative Error: {final_metrics['relative_error']:.4f}")

    elapsed = perf_counter() - start_time
    log(f"训练耗时: {str(timedelta(seconds=int(elapsed)))}")

    return model


if __name__ == '__main__':
    # 指定GPU 7
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'

    # 使用MILP_polished的数据
    data_dir = '/home/jjw/milp_yxy/MILP_polished/data/samples/setcov/500r'
    train_files = sorted(glob.glob(f'{data_dir}/train_set_level/sample_*.pkl'))
    valid_files = sorted(glob.glob(f'{data_dir}/valid_set_level/sample_*.pkl'))

    print(f"找到 {len(train_files)} 个训练样本")
    print(f"找到 {len(valid_files)} 个验证样本")
    print(f"使用GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'auto')}")
    print()

    if len(train_files) == 0:
        print("[ERROR] 未找到训练数据！")
        exit(1)

    # 训练模型
    save_dir = '/home/jjw/milp_yxy/MILP_set_1222/experiments/results/setcov_500_direct_topk'

    model = train_model(
        train_files=train_files,
        valid_files=valid_files,
        save_dir=save_dir,
        emb_size=32,
        n_layers=3,
        max_epochs=100,
        lr=1e-4,
        lambda_value=1.0,
        lambda_selection=2.0,  # ✅ 从0.5提升到2.0，强化排序学习
        patience=10,
        early_stopping=20
    )

    print("\n[SUCCESS] DirectTopKSelector训练完成！")
