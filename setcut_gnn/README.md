# MILP_set_1222: 直接Top-K割平面选择模型

## 核心改进

### 问题：原MILP_polished的矛盾

**理论矛盾**：
- 论文证明：割平面下界提升函数是**非次模的**（non-submodular）
- 原推理方法：使用**贪心搜索**（假设次模性）
- 结论：方法与理论矛盾

**贪心搜索的问题**：
```python
# 原方法：迭代贪心
selected = []
for _ in range(K):
    # 每次选择边际贡献最大的一个
    best_c = argmax_{c not in selected} Δz(selected ∪ {c}) - Δz(selected)
    selected.append(best_c)
```

这假设了边际贡献递减（次模性），但实际上边际贡献可能递增！

### 解决方案：直接Top-K选择

**新方法**：
1. 模型接收**所有N个**候选割平面
2. 直接输出：
   - Top-K个割平面的索引
   - 这K个割平面的联合下界提升
   - 每个割平面对联合下界提升的贡献（用于排序）

**理论一致性**：
- 不依赖贪心假设
- 模型学习的是全局选择策略
- 符合QUBO理论框架

## 模型架构

### DirectTopKSelector

```
输入：
  - state: 异构图（变量、约束、N个割平面）
  - K: 目标选择数量

流程：
  1. GNN编码 → cut_embeddings [N, emb_size]
  2. 选择头 → selection_scores [N]
  3. Top-K选择 → top_k_indices [K]
  4. 集合值预测 → joint_improvement scalar

输出：
  - top_k_indices: [K] 选中的割平面索引
  - selection_scores: [N] 每个割平面的选择分数
  - joint_improvement: scalar 预测的联合下界提升
```

### 训练流程

**数据收集**：
- 与MILP_polished相同
- 每个样本：M个子集 + 联合下界提升

**训练目标**：
```python
L = L_selection + λ * L_value

# L_selection: 选择损失
# 确保真正优秀的割平面得分高
# 方法1: Ranking loss
# 方法2: Cross-entropy with labels

# L_value: 值预测损失
# MSE + Contrastive loss（与MILP_polished相同）
```

**推理**：
```python
# 直接一次前向传播
top_k_indices, scores, value = model(state, K=10)
# 无需迭代贪心！
```

## 关键创新

1. **理论一致性**：不依赖次模假设
2. **效率提升**：一次前向传播 vs K次迭代
3. **端到端学习**：直接优化Top-K选择目标
4. **可解释性**：selection_scores提供每个割平面的重要性

## 实验对比

对比项 | MILP_polished | MILP_set_1222
-------|--------------|---------------
推理方法 | 贪心搜索（K次迭代） | 直接Top-K（一次）
理论依据 | ❌ 假设次模性 | ✅ 非次模理论
推理时间 | O(K × N) | O(1)
训练目标 | 集合值预测 | 选择+值联合优化
可解释性 | 低（迭代过程） | 高（显式选择分数）

## 文件结构

```
MILP_set_1222/
├── data/
│   └── collector.py          # 数据收集（复用MILP_polished）
├── models/
│   ├── base_gnn.py           # 基础GNN编码器
│   ├── selection_head.py     # 选择头
│   └── direct_topk_model.py  # 完整模型
├── utils/
│   ├── losses.py             # 损失函数
│   └── graph_builder.py      # 图构建
├── experiments/
│   ├── train.py              # 训练脚本
│   └── evaluate.py           # 评估脚本
└── README.md
```

## TODO

- [ ] 实现 DirectTopKSelector 模型
- [ ] 设计选择损失函数
- [ ] 训练模型
- [ ] 对比评估（vs 贪心、vs MILP_polished）
- [ ] 分析选择分数的可解释性
