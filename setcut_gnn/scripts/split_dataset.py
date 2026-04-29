"""
数据集分割脚本

将5000个样本分成:
- 4000个训练样本
- 500个验证样本
- 500个测试样本
"""

import os
import shutil
import random
from glob import glob

def split_dataset(data_dir, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1, seed=42):
    """
    分割数据集

    Args:
        data_dir: 数据目录
        train_ratio: 训练集比例 (0.8 = 4000/5000)
        valid_ratio: 验证集比例 (0.1 = 500/5000)
        test_ratio: 测试集比例 (0.1 = 500/5000)
        seed: 随机种子
    """
    random.seed(seed)

    # 源目录和目标目录
    src_dir = os.path.join(data_dir, 'train_set_level')
    train_dir = os.path.join(data_dir, 'train_set_level_split')
    valid_dir = os.path.join(data_dir, 'valid_set_level')
    test_dir = os.path.join(data_dir, 'test_set_level')

    # 获取所有样本文件
    sample_files = sorted(glob(os.path.join(src_dir, 'sample_*.pkl')))
    n_samples = len(sample_files)

    print(f"找到 {n_samples} 个样本文件")

    if n_samples == 0:
        print("[ERROR] 没有找到样本文件!")
        return

    # 随机打乱
    random.shuffle(sample_files)

    # 计算分割点
    n_train = int(n_samples * train_ratio)
    n_valid = int(n_samples * valid_ratio)
    n_test = n_samples - n_train - n_valid

    print(f"分割方案: 训练={n_train}, 验证={n_valid}, 测试={n_test}")

    # 分割
    train_files = sample_files[:n_train]
    valid_files = sample_files[n_train:n_train + n_valid]
    test_files = sample_files[n_train + n_valid:]

    # 创建目录
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # 复制文件
    def copy_files(files, dst_dir, prefix):
        for i, f in enumerate(files, 1):
            dst = os.path.join(dst_dir, f'sample_{i}.pkl')
            shutil.copy2(f, dst)
            if i % 500 == 0:
                print(f"  {prefix}: {i}/{len(files)}")
        print(f"  {prefix}: 完成 {len(files)} 个文件")

    print("\n开始复制文件...")
    copy_files(train_files, train_dir, "训练集")
    copy_files(valid_files, valid_dir, "验证集")
    copy_files(test_files, test_dir, "测试集")

    # 重命名原目录，使用新的训练目录
    backup_dir = os.path.join(data_dir, 'train_set_level_backup')
    if os.path.exists(src_dir) and not os.path.exists(backup_dir):
        os.rename(src_dir, backup_dir)
        os.rename(train_dir, src_dir)
        print(f"\n原目录已备份到: {backup_dir}")
        print(f"新训练集目录: {src_dir}")

    print("\n=== 分割完成 ===")
    print(f"训练集: {len(train_files)} 个样本 -> {src_dir}")
    print(f"验证集: {len(valid_files)} 个样本 -> {valid_dir}")
    print(f"测试集: {len(test_files)} 个样本 -> {test_dir}")


if __name__ == '__main__':
    data_dir = '/home/jjw/milp_yxy/MILP_set_1222/data/samples/setcov/500r'
    split_dataset(data_dir, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1, seed=42)
