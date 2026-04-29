#!/usr/bin/env python
"""
重新收集数据样本 - 使用30维割平面特征

这个脚本会：
1. 使用现有的MILP实例
2. 调用collector收集新的样本（30维特征）
3. 保存到新目录
"""

import os
import sys
import glob
import numpy as np

sys.path.insert(0, '/home/jjw/milp_yxy/MILP_set_1222')

from data.collector import collect_set_level_samples

def main():
    # 数据目录
    instance_dir = '/home/jjw/milp_yxy/MILP_polished/data/instances/setcov/train_500r'

    # 查找实例
    instances = sorted(glob.glob(f'{instance_dir}/*.lp'))

    if len(instances) == 0:
        print(f"[ERROR] 未找到实例文件: {instance_dir}")
        return

    print(f"找到 {len(instances)} 个MILP实例")

    # 输出目录
    out_dir = '/home/jjw/milp_yxy/MILP_set_1222/data/samples/setcov/500r_30dim'

    # 收集样本
    rng = np.random.default_rng(42)

    # 训练集
    print("\n=== 收集训练集 ===")
    collect_set_level_samples(
        instances=instances,
        n_samples=4000,
        n_jobs=8,
        out_dir=f'{out_dir}/train_set_level',
        rng=rng
    )

    # 验证集
    print("\n=== 收集验证集 ===")
    collect_set_level_samples(
        instances=instances,
        n_samples=500,
        n_jobs=8,
        out_dir=f'{out_dir}/valid_set_level',
        rng=rng
    )

    # 测试集
    print("\n=== 收集测试集 ===")
    collect_set_level_samples(
        instances=instances,
        n_samples=500,
        n_jobs=8,
        out_dir=f'{out_dir}/test_set_level',
        rng=rng
    )

    print("\n[SUCCESS] 数据收集完成！")


if __name__ == '__main__':
    main()
