# MIQP 实验规划

目标：构建与 `milp_gnn.tex` 同级别的实验章节，补充MIQP特性（线性化误差、原始问题有效性等）。

## 数据阶段
1. 使用 `scripts/generate_instances.py` 生成 Portfolio/QAP/QCQP/BoxQP 实例。
2. `scripts/linearize_instance.py` 执行 PLA/McCormick/自适应线性化，记录误差界与规模膨胀比。
3. `scripts/extract_cuts.py` 收集根节点割平面，采样割平面集合并计算 `Δz^{MILP}`、`Δz^{MIQP}`、`ε`。

## 模型阶段
1. `scripts/build_graph_dataset.py` 生成四部图特征数据集。
2. `scripts/train_miqp_gnn.py` 训练层次感知HGT（含误差编码、Q注意力、辅助变量聚合）。
3. `scripts/eval_miqp_gnn.py` 输出 Gap闭合、原始问题有效性、线性化误差等指标。

## 报告阶段
1. `scripts/run_experiments.py` 驱动六组实验，产出 JSON/CSV。
2. 在 `docs/` 中补写：
   - 线性化方法对比表
   - 割平面选择性能表
   - 端到端求解性能表
   - 消融图表、泛化结果、效率分析
3. 将关键图表和表格嵌入 `miqp_gnn.tex`，格式参考 `milp_gnn.tex`。

action items:
- [ ] 实现实例生成与线性化细节
- [ ] 接入求解器API并输出真实割平面
- [ ] 完成数据管线与GNN训练脚本
- [ ] 积累实验结果并撰写报告

## 当前进展（占位版）
- [x] 手动创建Portfolio示例实例，生成`train_pla.npz`占位特征（变量/约束/Q密度）。
- [x] 训练脚本可读取NPZ并输出统计日志（待替换为真实HGT训练）。
- [ ] 补充有效的割平面特征、误差标签与评估脚本。
