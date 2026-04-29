"""
图构建工具 - 完整30维割平面特征版本

严格按照论文Table 3.5实现割平面节点特征：
- 违反度 (3维): violation, rel_violation, log_violation
- 深度 (3维): efficacy, dcd, normalized_efficacy
- 方向 (4维): obj_parallelism, expected_improvement, obj_parallelism_abs, angle_to_obj
- 稀疏性 (4维): support, int_support, nnz_count, density_rank
- 系数统计 (8维): coef_l1, coef_l2, coef_linf, coef_mean, coef_std, coef_max, coef_min, rhs_normalized
- 数值稳定性 (8维): dynamic_range, log_dynamic_range, max_coef_ratio, rhs_abs, rhs_sign, coef_int_ratio, is_gomory, cut_type
"""

import numpy as np
import scipy.sparse as sp
from math import floor


def get_state(model, cuts):
    """提取MILP问题的图表示（含完整30维割平面特征）

    Args:
        model: SCIP模型
        cuts: 候选割平面列表

    Returns:
        state: dict 包含图的完整状态
    """
    # 计算目标函数系数向量和范数
    obj_terms = model.getObjective().terms
    obj_coeffs = np.zeros(len(model.getLPColsData()))
    for var, coef in obj_terms.items():
        try:
            idx = var.getCol().getLPPos()
            if 0 <= idx < len(obj_coeffs):
                obj_coeffs[idx] = coef
        except:
            pass

    obj_norm = np.linalg.norm(obj_coeffs)
    obj_norm = 1.0 if obj_norm <= 0 else obj_norm

    # 获取LP行和列数据
    rows = model.getLPRowsData()
    cols = model.getLPColsData()
    n_rows = len(rows)
    n_cols = len(cols)
    n_cuts = len(cuts)

    # 获取整数变量索引
    int_var_indices = set()
    for i, col in enumerate(cols):
        vtype = col.getVar().vtype()
        if vtype in ['BINARY', 'INTEGER', 'IMPLINT']:
            int_var_indices.add(i)

    #=============== 约束（行）特征 ===============
    row_feats = {}

    row_norms = np.array([row.getNorm() for row in rows])
    row_norms[row_norms == 0] = 1

    lhs = np.array([row.getLhs() for row in rows])
    rhs = np.array([row.getRhs() for row in rows])
    has_lhs = [not model.isInfinity(-val) for val in lhs]
    has_rhs = [not model.isInfinity(val) for val in rhs]
    rows = np.array(rows)

    row_feats['rhs'] = np.concatenate((-(lhs / row_norms)[has_lhs], (rhs / row_norms)[has_rhs])).reshape(-1, 1)

    row_feats['is_tight'] = np.concatenate(
        ([row.getBasisStatus() == 'lower' for row in rows[has_lhs]],
         [row.getBasisStatus() == 'upper' for row in rows[has_rhs]])
    ).reshape(-1, 1)

    duals = np.array([model.getRowDualSol(row) for row in rows]) / (row_norms * obj_norm)
    row_feats['dual'] = np.concatenate((-duals[has_lhs], duals[has_rhs])).reshape(-1, 1)

    row_feat_names = []
    for k, v in row_feats.items():
        if v.shape[1] == 1:
            row_feat_names.append(k)
        else:
            row_feat_names.extend([f'{k}_{i}' for i in range(v.shape[1])])

    row_feat_vals = np.concatenate(list(row_feats.values()), axis=-1)
    cons_feats = {'features': row_feat_names, 'values': row_feat_vals}

    #=============== 约束-变量边特征 ===============
    data = np.array([
        [rows[i].getVals()[j] / row_norms[i], rows[i].getLPPos(), rows[i].getCols()[j].getLPPos()]
        for i in range(n_rows)
        for j in range(len(rows[i].getCols()))
    ])

    coef_matrix = sp.csr_matrix((data[:, 0], (data[:, 1], data[:, 2])), shape=(n_rows, n_cols))
    coef_matrix = sp.vstack((-coef_matrix[has_lhs, :], coef_matrix[has_rhs, :])).tocoo(copy=False)

    row_ind, col_ind = coef_matrix.row, coef_matrix.col

    cons_edge_feats = {
        'features': ['coef'],
        'indices': np.vstack([row_ind, col_ind]),
        'values': coef_matrix.data.reshape(-1, 1)
    }

    #=============== 变量（列）特征 ===============
    col_feats = {}

    type_map = {'BINARY': 0, 'INTEGER': 1, 'IMPLINT': 2, 'CONTINUOUS': 3}
    types = np.array([type_map[col.getVar().vtype()] for col in cols])
    col_feats['type'] = np.zeros((n_cols, 4))
    col_feats['type'][np.arange(n_cols), types] = 1

    col_feats['obj_coef'] = np.array([col.getObjCoeff() for col in cols]).reshape(-1, 1) / obj_norm

    lb = np.array([col.getLb() for col in cols])
    ub = np.array([col.getUb() for col in cols])
    has_lb = [not model.isInfinity(-val) for val in lb]
    has_ub = [not model.isInfinity(val) for val in ub]
    col_feats['has_lb'] = np.array(has_lb).astype(int).reshape(-1, 1)
    col_feats['has_ub'] = np.array(has_ub).astype(int).reshape(-1, 1)
    col_feats['at_lb'] = np.array([col.getBasisStatus() == 'lower' for col in cols]).reshape(-1, 1)
    col_feats['at_ub'] = np.array([col.getBasisStatus() == 'upper' for col in cols]).reshape(-1, 1)

    col_feats['frac'] = np.array(
        [0.5 - abs(col.getVar().getLPSol() - floor(col.getVar().getLPSol()) - 0.5) for col in cols]
    ).reshape(-1, 1)
    col_feats['frac'][types == 3] = 0

    col_feats['reduced_cost'] = np.array([model.getVarRedcost(col.getVar()) for col in cols]).reshape(-1, 1) / obj_norm

    col_feats['lp_val'] = np.array([col.getVar().getLPSol() for col in cols]).reshape(-1, 1)

    sols = model.getSols()
    if len(sols) != 0:
        incumbent = model.getBestSol()
        col_feats['primal_val'] = np.array([model.getSolVal(incumbent, col.getVar()) for col in cols]).reshape(-1, 1)
        col_feats['avg_primal'] = np.mean([
            [model.getSolVal(sol, col.getVar()) for sol in sols] for col in cols
        ], axis=1).reshape(-1, 1)
    else:
        col_feats['primal_val'] = np.zeros(n_cols).reshape(-1, 1)
        col_feats['avg_primal'] = np.zeros(n_cols).reshape(-1, 1)

    col_feat_names = []
    for k, v in col_feats.items():
        if v.shape[1] == 1:
            col_feat_names.append(k)
        else:
            col_feat_names.extend([f'{k}_{i}' for i in range(v.shape[1])])

    col_feat_vals = np.concatenate(list(col_feats.values()), axis=-1)
    var_feats = {'features': col_feat_names, 'values': col_feat_vals}

    #=============== 割平面特征 (30维) ===============
    cut_feats = _compute_cut_features_30d(model, cuts, cols, obj_coeffs, obj_norm, int_var_indices, n_cols)

    cut_feat_names = cut_feats['feature_names']
    cut_feat_vals = cut_feats['values']

    #=============== 割平面-变量边特征 ===============
    cut_norms = np.array([cut.getNorm() for cut in cuts])
    cut_norms[cut_norms == 0] = 1

    cut_data = []
    for i, cut in enumerate(cuts):
        cols_in_cut = cut.getCols()
        vals_in_cut = cut.getVals()
        for j, (col, val) in enumerate(zip(cols_in_cut, vals_in_cut)):
            cut_data.append([val / cut_norms[i], i, col.getLPPos()])

    if len(cut_data) > 0:
        cut_data = np.array(cut_data)
        cut_coef_matrix = sp.csr_matrix(
            (cut_data[:, 0], (cut_data[:, 1], cut_data[:, 2])),
            shape=(n_cuts, n_cols)
        ).tocoo(copy=False)

        cut_edge_feats = {
            'features': ['coef'],
            'indices': np.vstack([cut_coef_matrix.row, cut_coef_matrix.col]),
            'values': cut_coef_matrix.data.reshape(-1, 1)
        }
    else:
        cut_edge_feats = {
            'features': ['coef'],
            'indices': np.zeros((2, 0), dtype=int),
            'values': np.zeros((0, 1))
        }

    # 构建最终状态
    state = {
        'constraint_features': cons_feats['values'],
        'constraint_feat_names': cons_feats['features'],

        'variable_features': var_feats['values'],
        'variable_feat_names': var_feats['features'],

        'cut_features': cut_feat_vals,
        'cut_feat_names': cut_feat_names,

        'var_cons_edges': cons_edge_feats['indices'],
        'var_cons_edge_features': cons_edge_feats['values'],
        'var_cons_edge_feat_names': cons_edge_feats['features'],

        'var_cut_edges': cut_edge_feats['indices'],
        'var_cut_edge_features': cut_edge_feats['values'],
        'var_cut_edge_feat_names': cut_edge_feats['features'],

        'n_vars': var_feats['values'].shape[0],
        'n_cons': cons_feats['values'].shape[0],
        'n_cuts': cut_feat_vals.shape[0]
    }

    return state


def _compute_cut_features_30d(model, cuts, cols, obj_coeffs, obj_norm, int_var_indices, n_cols):
    """
    计算割平面的30维特征（严格按照论文Table 3.5）

    特征分类：
    1. 违反度 (3维): violation, rel_violation, log_violation
    2. 深度 (3维): efficacy, dcd, normalized_efficacy
    3. 方向 (4维): obj_parallelism, expected_improvement, obj_parallelism_abs, angle_to_obj
    4. 稀疏性 (4维): support, int_support, nnz_count, density_rank
    5. 系数统计 (8维): coef_l1, coef_l2, coef_linf, coef_mean, coef_std, coef_max, coef_min, rhs_normalized
    6. 数值稳定性 (8维): dynamic_range, log_dynamic_range, max_coef_ratio, rhs_abs, rhs_sign, coef_int_ratio, is_gomory, cut_type

    Args:
        model: SCIP模型
        cuts: 割平面列表
        cols: LP列数据
        obj_coeffs: 目标函数系数向量
        obj_norm: 目标函数范数
        int_var_indices: 整数变量索引集合
        n_cols: 变量总数

    Returns:
        dict: {'feature_names': list, 'values': ndarray}
    """
    n_cuts = len(cuts)

    if n_cuts == 0:
        return {
            'feature_names': _get_cut_feature_names(),
            'values': np.zeros((0, 30), dtype=np.float32)
        }

    # 预计算
    cut_norms = np.array([cut.getNorm() for cut in cuts])
    cut_norms[cut_norms == 0] = 1.0

    # 获取LP解
    lp_sol = np.array([col.getVar().getLPSol() for col in cols])

    # 初始化30维特征数组
    features = np.zeros((n_cuts, 30), dtype=np.float32)

    # 为每个割平面计算特征
    efficacy_list = []

    for i, cut in enumerate(cuts):
        # 获取割平面系数
        cut_cols = cut.getCols()
        cut_vals = cut.getVals()

        # 构建系数向量
        alpha = np.zeros(n_cols, dtype=np.float64)
        for col, val in zip(cut_cols, cut_vals):
            idx = col.getLPPos()
            if 0 <= idx < n_cols:
                alpha[idx] = val

        # RHS
        lhs = cut.getLhs()
        rhs = cut.getRhs()
        has_lhs = not model.isInfinity(-lhs)
        has_rhs = not model.isInfinity(rhs)

        # 选择活跃侧
        activity = model.getRowActivity(cut)
        if has_lhs and has_rhs:
            if (lhs - activity) > (activity - rhs):
                gamma = lhs
                sign = -1
            else:
                gamma = rhs
                sign = 1
        elif has_lhs:
            gamma = lhs
            sign = -1
        else:
            gamma = rhs
            sign = 1

        alpha_signed = sign * alpha
        norm = cut_norms[i]

        # ========== 1. 违反度 (3维) ==========
        # violation: f_vio = α^T x* - γ
        violation = np.dot(alpha_signed, lp_sol) - gamma
        features[i, 0] = violation

        # rel_violation: f_rv = f_vio / max(|γ|, 1)
        rel_violation = violation / max(abs(gamma), 1.0)
        features[i, 1] = rel_violation

        # log_violation: log(1 + max(f_vio, 0))
        log_violation = np.log1p(max(violation, 0))
        features[i, 2] = log_violation

        # ========== 2. 深度 (3维) ==========
        # efficacy: f_eff = f_vio / ||α||_2
        efficacy = violation / norm if norm > 0 else 0
        features[i, 3] = efficacy
        efficacy_list.append(efficacy)

        # dcd: 有向截断距离（需要最优整数解，这里用近似）
        # 使用SCIP的getCutEfficacy作为替代
        try:
            dcd = model.getCutEfficacy(cut)
        except:
            dcd = efficacy
        features[i, 4] = dcd

        # normalized_efficacy: 稍后归一化
        features[i, 5] = efficacy  # 临时存储，稍后归一化

        # ========== 3. 方向 (4维) ==========
        # obj_parallelism: f_op = α^T c / (||α||_2 ||c||_2)
        alpha_dot_c = np.dot(alpha_signed, obj_coeffs)
        obj_parallelism = alpha_dot_c / (norm * obj_norm) if (norm > 0 and obj_norm > 0) else 0
        features[i, 6] = obj_parallelism

        # expected_improvement: f_ei = f_op * f_eff
        expected_improvement = obj_parallelism * efficacy
        features[i, 7] = expected_improvement

        # obj_parallelism_abs: |f_op|
        features[i, 8] = abs(obj_parallelism)

        # angle_to_obj: arccos(|f_op|)
        features[i, 9] = np.arccos(np.clip(abs(obj_parallelism), 0, 1))

        # ========== 4. 稀疏性 (4维) ==========
        # 非零系数
        nonzero_mask = alpha != 0
        nnz = np.sum(nonzero_mask)

        # support: |supp(α)| / n
        features[i, 10] = nnz / n_cols if n_cols > 0 else 0

        # int_support: |supp(α) ∩ I| / |supp(α)|
        int_nnz = sum(1 for j in range(n_cols) if nonzero_mask[j] and j in int_var_indices)
        features[i, 11] = int_nnz / nnz if nnz > 0 else 0

        # nnz_count: 非零系数个数（归一化）
        features[i, 12] = nnz / n_cols if n_cols > 0 else 0

        # density_rank: 稀疏性排名（稍后计算）
        features[i, 13] = nnz  # 临时存储原始值

        # ========== 5. 系数统计 (8维) ==========
        if nnz > 0:
            nonzero_vals = alpha[nonzero_mask]

            # coef_l1: ||α||_1
            coef_l1 = np.sum(np.abs(nonzero_vals))
            features[i, 14] = np.log1p(coef_l1)  # 对数变换

            # coef_l2: ||α||_2
            features[i, 15] = np.log1p(norm)

            # coef_linf: ||α||_∞
            coef_linf = np.max(np.abs(nonzero_vals))
            features[i, 16] = np.log1p(coef_linf)

            # coef_mean: mean(|α|)
            features[i, 17] = np.mean(np.abs(nonzero_vals))

            # coef_std: std(|α|)
            features[i, 18] = np.std(np.abs(nonzero_vals))

            # coef_max: max(α)
            features[i, 19] = np.max(nonzero_vals)

            # coef_min: min(α)
            features[i, 20] = np.min(nonzero_vals)

            # rhs_normalized: γ / ||α||_2
            features[i, 21] = gamma / norm if norm > 0 else 0
        else:
            features[i, 14:22] = 0

        # ========== 6. 数值稳定性 (8维) ==========
        if nnz > 0:
            abs_nonzero = np.abs(nonzero_vals)
            min_abs = np.min(abs_nonzero)

            # dynamic_range: max|α_j| / min_{j:α_j≠0}|α_j|
            dynamic_range = coef_linf / min_abs if min_abs > 1e-10 else 1e10
            features[i, 22] = min(dynamic_range, 1e10)

            # log_dynamic_range
            features[i, 23] = np.log1p(min(dynamic_range, 1e10))

            # max_coef_ratio: ||α||_∞ / ||α||_1
            features[i, 24] = coef_linf / coef_l1 if coef_l1 > 0 else 0
        else:
            features[i, 22:25] = 0

        # rhs_abs: |γ|
        features[i, 25] = np.log1p(abs(gamma))

        # rhs_sign: sign(γ)
        features[i, 26] = np.sign(gamma)

        # coef_int_ratio: 整数系数占比
        if nnz > 0:
            int_coef_count = sum(1 for v in nonzero_vals if abs(v - round(v)) < 1e-6)
            features[i, 27] = int_coef_count / nnz
        else:
            features[i, 27] = 0

        # is_gomory: 是否为Gomory割（基于名称判断）
        try:
            cut_name = cut.getName().lower() if hasattr(cut, 'getName') else ""
            is_gomory = 1.0 if 'gomory' in cut_name or 'gmi' in cut_name else 0.0
        except:
            is_gomory = 0.0
        features[i, 28] = is_gomory

        # cut_type: 割平面类型编码
        # 0: unknown, 1: gomory, 2: mir, 3: flowcover, 4: clique, 5: other
        try:
            cut_name = cut.getName().lower() if hasattr(cut, 'getName') else ""
            if 'gomory' in cut_name or 'gmi' in cut_name:
                cut_type = 1
            elif 'mir' in cut_name:
                cut_type = 2
            elif 'flow' in cut_name:
                cut_type = 3
            elif 'clique' in cut_name:
                cut_type = 4
            else:
                cut_type = 5
        except:
            cut_type = 0
        features[i, 29] = cut_type / 5.0  # 归一化

    # 后处理：归一化需要全局信息的特征
    if n_cuts > 0:
        # normalized_efficacy: f_eff / max_{c' in C} f_eff(c')
        max_efficacy = max(efficacy_list) if efficacy_list else 1.0
        max_efficacy = max_efficacy if max_efficacy > 0 else 1.0
        features[:, 5] = features[:, 5] / max_efficacy

        # density_rank: 稀疏性排名（归一化到0-1）
        nnz_values = features[:, 13].copy()
        ranks = np.argsort(np.argsort(nnz_values))
        features[:, 13] = ranks / max(n_cuts - 1, 1)

    # 处理NaN和Inf
    features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)

    return {
        'feature_names': _get_cut_feature_names(),
        'values': features.astype(np.float32)
    }


def _get_cut_feature_names():
    """返回30维割平面特征名称列表"""
    return [
        # 违反度 (3维)
        'violation',
        'rel_violation',
        'log_violation',
        # 深度 (3维)
        'efficacy',
        'dcd',
        'normalized_efficacy',
        # 方向 (4维)
        'obj_parallelism',
        'expected_improvement',
        'obj_parallelism_abs',
        'angle_to_obj',
        # 稀疏性 (4维)
        'support',
        'int_support',
        'nnz_count',
        'density_rank',
        # 系数统计 (8维)
        'coef_l1',
        'coef_l2',
        'coef_linf',
        'coef_mean',
        'coef_std',
        'coef_max',
        'coef_min',
        'rhs_normalized',
        # 数值稳定性 (8维)
        'dynamic_range',
        'log_dynamic_range',
        'max_coef_ratio',
        'rhs_abs',
        'rhs_sign',
        'coef_int_ratio',
        'is_gomory',
        'cut_type'
    ]
