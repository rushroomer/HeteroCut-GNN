"""
训练脚本 - 使用Quality@K损失函数

核心改进：
1. Quality@K作为主要学习目标（目标>=1.1）
2. 完整的训练日志，显示每个K值的Quality@K变化
3. 排序损失作为次要目标
4. 支持30维割平面特征
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

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.direct_topk_model import DirectTopKSelector
from utils.quality_at_k_loss import combined_quality_and_value_loss
from utils.losses import compute_cut_qualities, combined_ranking_loss


def prepare_batch(sample_files):
    """生成训练批次"""
    for fname in sample_files:
        # 检查是否是gzip压缩文件
        with open(fname, 'rb') as f:
            magic = f.read(2)
            f.seek(0)
            if magic == b'\x1f\x8b':  # gzip magic number
                with gzip.open(fname, 'rb') as gf:
                    data = pickle.load(gf)
            else:
                data = pickle.load(f)

        state = data['state']
        subsets_data = data['subsets']

        # 转换state为tensor格式
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


# 注意：去掉@tf.function装饰器，因为compute_efficacy_topk中有numpy操作
# 这会让训练稍慢，但避免了tf.function与numpy的兼容性问题
def train_step(model, optimizer, state, subsets_data,
               lambda_quality=2.0, lambda_value=1.0,
               K_values=[5, 10, 20], target_ratio=2.0):
    """
    训练步骤 - 使用Quality@K损失

    Args:
        lambda_quality: float Quality@K损失权重
        lambda_value: float Value损失权重
        K_values: List[int] 优化的K值列表
        target_ratio: float 目标Quality@K（2.0 = 超过Efficacy 100%）
    """
    with tf.GradientTape() as tape:
        # 组合损失：Quality@K + Value
        total_loss, loss_dict = combined_quality_and_value_loss(
            model, state, subsets_data,
            lambda_quality=lambda_quality,
            lambda_value=lambda_value,
            K_values=K_values,
            target_ratio=target_ratio
        )

    # 反向传播
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return total_loss, loss_dict


def evaluate(model, sample_files, lambda_quality=2.0, lambda_value=1.0,
             K_values=[5, 10, 20], target_ratio=2.0):
    """评估模型"""
    metrics = {
        'loss': [],
        'quality': [],
        'value': [],
        'quality_k5': [],
        'quality_k10': [],
        'quality_k20': []
    }

    for state, subsets_data in prepare_batch(sample_files):
        try:
            # 计算损失
            total_loss, loss_dict = combined_quality_and_value_loss(
                model, state, subsets_data,
                lambda_quality=lambda_quality,
                lambda_value=lambda_value,
                K_values=K_values,
                target_ratio=target_ratio
            )

            metrics['loss'].append(total_loss.numpy())
            metrics['quality'].append(loss_dict['quality'])
            metrics['value'].append(loss_dict['value'])

            # 记录各K的Quality@K
            if 'K=5' in loss_dict:
                metrics['quality_k5'].append(loss_dict['K=5']['quality_k'])
            if 'K=10' in loss_dict:
                metrics['quality_k10'].append(loss_dict['K=10']['quality_k'])
            if 'K=20' in loss_dict:
                metrics['quality_k20'].append(loss_dict['K=20']['quality_k'])

        except Exception as e:
            print(f"[WARNING] Evaluation failed for sample: {e}")
            continue

    # 平均
    return {k: np.mean(v) if v else 0.0 for k, v in metrics.items()}


def train_model(train_files, valid_files, save_dir,
                emb_size=32, n_layers=3, max_epochs=100,
                lr=1e-4, lambda_quality=5.0, lambda_value=1.0,
                lambda_selection=0.5,  # 新增：选择排序损失权重
                K_values=[5, 10, 20], target_ratio=1.1,  # 修改：目标1.1
                patience=10, early_stopping=20):
    """
    训练DirectTopKSelector模型 - 使用Quality@K损失

    Args:
        lambda_quality: float Quality@K损失权重（建议2.0-5.0）
        lambda_value: float Value损失权重
        lambda_selection: float 选择排序损失权重（次要目标）
        K_values: List[int] 优化的K值列表
        target_ratio: float 目标Quality@K（1.1 = 超过Efficacy 10%）
        其他参数同原train.py
    """
    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, 'training.log')
    metrics_file = os.path.join(save_dir, 'metrics_history.pkl')

    def log(msg):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        formatted = f"[{timestamp}] {msg}"
        print(formatted)
        with open(log_file, 'a') as f:
            f.write(formatted + '\n')

    log("=" * 80)
    log("DirectTopKSelector模型训练 - Quality@K优化版")
    log("=" * 80)
    log(f"训练样本数: {len(train_files)}")
    log(f"验证样本数: {len(valid_files)}")
    log(f"嵌入维度: {emb_size}")
    log(f"GNN 层数: {n_layers}")
    log(f"学习率: {lr}")
    log(f"Quality@K损失权重: {lambda_quality}")
    log(f"Value损失权重: {lambda_value}")
    log(f"Selection损失权重: {lambda_selection}")
    log(f"优化的K值: {K_values}")
    log(f"目标Quality@K: {target_ratio}x (超过Efficacy {(target_ratio-1)*100:.0f}%)")
    log("")

    # 创建模型
    model = DirectTopKSelector(emb_size=emb_size, n_layers=n_layers)
    optimizer = Adam(learning_rate=lr)

    best_quality_score = -np.inf  # 综合Quality分数
    plateau_count = 0
    no_improve_count = 0
    start_time = perf_counter()

    # 训练历史
    history = {
        'epochs': [],
        'train_loss': [],
        'valid_loss': [],
        'quality_at_5': [],
        'quality_at_10': [],
        'quality_at_20': []
    }

    for epoch in range(1, max_epochs + 1):
        log(f"\n{'='*60}")
        log(f"[Epoch {epoch}/{max_epochs}]")
        log('='*60)

        # 训练
        train_metrics = {
            'loss': [],
            'quality': [],
            'value': [],
            'quality_k5': [],
            'quality_k10': [],
            'quality_k20': []
        }

        # 随机打乱训练集
        np.random.shuffle(train_files)

        for i, (state, subsets_data) in enumerate(prepare_batch(train_files)):
            try:
                loss, loss_dict = train_step(
                    model, optimizer, state, subsets_data,
                    lambda_quality, lambda_value,
                    K_values, target_ratio
                )

                train_metrics['loss'].append(loss.numpy())
                train_metrics['quality'].append(loss_dict['quality'])
                train_metrics['value'].append(loss_dict['value'])

                # 记录各K的Quality@K
                if 'K=5' in loss_dict:
                    train_metrics['quality_k5'].append(loss_dict['K=5']['quality_k'])
                if 'K=10' in loss_dict:
                    train_metrics['quality_k10'].append(loss_dict['K=10']['quality_k'])
                if 'K=20' in loss_dict:
                    train_metrics['quality_k20'].append(loss_dict['K=20']['quality_k'])

                if (i + 1) % 100 == 0:
                    recent_q5 = np.mean(train_metrics['quality_k5'][-100:]) if train_metrics['quality_k5'] else 0
                    recent_q10 = np.mean(train_metrics['quality_k10'][-100:]) if train_metrics['quality_k10'] else 0
                    recent_q20 = np.mean(train_metrics['quality_k20'][-100:]) if train_metrics['quality_k20'] else 0
                    log(f"  [Train] Batch {i+1}/{len(train_files)} - "
                        f"Loss: {np.mean(train_metrics['loss'][-100:]):.4f}, "
                        f"Q@5: {recent_q5:.3f}, Q@10: {recent_q10:.3f}, Q@20: {recent_q20:.3f}")

            except Exception as e:
                log(f"  [WARNING] Batch {i} failed: {e}")
                continue

        # 平均训练指标
        avg_train = {k: np.mean(v) if v else 0.0 for k, v in train_metrics.items()}

        log(f"\n  [Train Summary]")
        log(f"    Loss: {avg_train['loss']:.4f} (Quality: {avg_train['quality']:.4f}, Value: {avg_train['value']:.4f})")

        # 验证
        valid_metrics = evaluate(
            model, valid_files,
            lambda_quality, lambda_value,
            K_values, target_ratio
        )

        log(f"\n  [Valid Summary]")
        log(f"    Loss: {valid_metrics['loss']:.4f}")

        # Quality@K详细输出（核心指标）
        log(f"\n  [Quality@K Results] *** 核心指标 ***")
        for K in K_values:
            k_name = f'quality_k{K}'
            qk = valid_metrics.get(k_name, 0.0)
            status = "✓ 达标" if qk >= target_ratio else "✗ 未达标"
            log(f"    Quality@{K}: {qk:.4f} {status} (目标: >={target_ratio})")

        # 记录历史
        history['epochs'].append(epoch)
        history['train_loss'].append(avg_train['loss'])
        history['valid_loss'].append(valid_metrics['loss'])
        history['quality_at_5'].append(valid_metrics.get('quality_k5', 0.0))
        history['quality_at_10'].append(valid_metrics.get('quality_k10', 0.0))
        history['quality_at_20'].append(valid_metrics.get('quality_k20', 0.0))

        # 综合Quality分数（加权）
        current_quality_score = (
            2.0 * valid_metrics.get('quality_k5', 0.0) +
            1.0 * valid_metrics.get('quality_k10', 0.0) +
            0.5 * valid_metrics.get('quality_k20', 0.0)
        ) / 3.5

        # 保存最佳模型
        if current_quality_score > best_quality_score:
            best_quality_score = current_quality_score
            plateau_count = 0
            no_improve_count = 0

            # 保存模型
            model_path = os.path.join(save_dir, 'best_model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model.get_weights(), f)

            # 保存详细信息
            best_info = {
                'epoch': epoch,
                'quality_score': current_quality_score,
                'quality_at_5': valid_metrics.get('quality_k5', 0.0),
                'quality_at_10': valid_metrics.get('quality_k10', 0.0),
                'quality_at_20': valid_metrics.get('quality_k20', 0.0),
                'loss': valid_metrics['loss']
            }
            with open(os.path.join(save_dir, 'best_model_info.pkl'), 'wb') as f:
                pickle.dump(best_info, f)

            log(f"\n  *** 最佳模型已保存 (Score: {current_quality_score:.4f}) ***")

        else:
            no_improve_count += 1
            plateau_count += 1

            # 学习率衰减
            if plateau_count >= patience:
                old_lr = optimizer.learning_rate.numpy()
                new_lr = old_lr * 0.5
                optimizer.learning_rate.assign(new_lr)
                log(f"\n  [学习率衰减] {old_lr:.2e} -> {new_lr:.2e}")
                plateau_count = 0

            # 早停
            if no_improve_count >= early_stopping:
                log(f"\n  [早停] {early_stopping} 轮无改进")
                break

        # 保存历史
        with open(metrics_file, 'wb') as f:
            pickle.dump(history, f)

    # 训练完成
    elapsed = perf_counter() - start_time
    log("\n" + "=" * 80)
    log("训练完成")
    log("=" * 80)
    log(f"训练耗时: {str(timedelta(seconds=int(elapsed)))}")
    log(f"最佳Quality Score: {best_quality_score:.4f}")

    # 加载最佳模型
    model_path = os.path.join(save_dir, 'best_model.pkl')
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model.set_weights(pickle.load(f))
        log("已加载最佳模型")

    # 最终评估
    log("\n=== 最终评估 ===")
    final_metrics = evaluate(model, valid_files, lambda_quality, lambda_value, K_values, target_ratio)

    all_passed = True
    for K in K_values:
        k_name = f'quality_k{K}'
        qk = final_metrics.get(k_name, 0.0)
        status = "✓ 达标" if qk >= target_ratio else "✗ 未达标"
        if qk < target_ratio:
            all_passed = False
        log(f"  Quality@{K}: {qk:.4f} {status}")

    if all_passed:
        log(f"\n*** 恭喜！所有Quality@K指标均达到目标 ({target_ratio}x) ***")
    else:
        log(f"\n*** 部分Quality@K指标未达标，建议继续调优 ***")

    return model


if __name__ == "__main__":
    import glob

    # 设置GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # 数据路径（使用MILP_set_1222的新数据）
    data_dir = "/home/jjw/milp_yxy/MILP_set_1222/data/samples/setcov/500r"
    train_dir = os.path.join(data_dir, "train_set_level")
    valid_dir = os.path.join(data_dir, "valid_set_level")

    train_files = sorted(glob.glob(os.path.join(train_dir, "*.pkl")))
    valid_files = sorted(glob.glob(os.path.join(valid_dir, "*.pkl")))

    print(f"找到 {len(train_files)} 个训练样本")
    print(f"找到 {len(valid_files)} 个验证样本")

    # 如果验证集为空，从训练集分割
    if len(valid_files) == 0:
        print("[WARNING] 验证集为空，使用训练集的10%作为验证集")
        n_valid = max(1, len(train_files) // 10)
        valid_files = train_files[-n_valid:]
        train_files = train_files[:-n_valid]
        print(f"分割后：训练={len(train_files)}, 验证={len(valid_files)}")

    if len(train_files) == 0:
        print("[ERROR] 未找到训练数据！请先运行数据收集脚本。")
        exit(1)

    # 保存目录
    save_dir = "/home/jjw/milp_yxy/MILP_set_1222/experiments/results/quality_at_k_v2"

    # 训练
    model = train_model(
        train_files,
        valid_files,
        save_dir,
        emb_size=64,       # 增大嵌入维度以支持30维特征
        n_layers=4,        # 增加层数
        max_epochs=100,
        lr=1e-4,
        lambda_quality=3.0,   # Quality@K主要目标
        lambda_value=1.0,     # Value辅助
        lambda_selection=0.5, # Selection次要目标
        K_values=[5, 10, 20],
        target_ratio=1.1,     # 目标：超过Efficacy 10%
        patience=10,
        early_stopping=30
    )

    print("\n[SUCCESS] Quality@K优化训练完成！")
    print(f"模型保存在: {save_dir}/best_model.pkl")
