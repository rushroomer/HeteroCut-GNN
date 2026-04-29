"""
图构建工具 - 简化稳定版本

基于原始MILP项目，但使用稳定的API调用
"""

import numpy as np
import scipy.sparse as sp
from math import floor


def get_state(model, cuts):
    """提取MILP问题的图表示

    Args:
        model: SCIP模型
        cuts: 候选割平面列表

    Returns:
        state: dict 包含图的完整状态
    """
    # 计算目标函数范数
    obj_norm = np.linalg.norm(list(model.getObjective().terms.values()))
    obj_norm = 1 if obj_norm <= 0 else obj_norm

    # 获取LP行和列数据
    rows = model.getLPRowsData()
    cols = model.getLPColsData()
    n_rows = len(rows)
    n_cols = len(cols)
    n_cuts = len(cuts)

    #=============== 约束（行）特征 ===============
    row_feats = {}

    # 计算每个约束的范数
    row_norms = np.array([row.getNorm() for row in rows])
    row_norms[row_norms == 0] = 1

    # 处理双边约束
    lhs = np.array([row.getLhs() for row in rows])
    rhs = np.array([row.getRhs() for row in rows])
    has_lhs = [not model.isInfinity(-val) for val in lhs]
    has_rhs = [not model.isInfinity(val) for val in rhs]
    rows = np.array(rows)

    # RHS特征
    row_feats['rhs'] = np.concatenate((-(lhs / row_norms)[has_lhs], (rhs / row_norms)[has_rhs])).reshape(-1, 1)

    # 紧度指示器
    row_feats['is_tight'] = np.concatenate(
        ([row.getBasisStatus() == 'lower' for row in rows[has_lhs]],
         [row.getBasisStatus() == 'upper' for row in rows[has_rhs]])
    ).reshape(-1, 1)

    # 对偶值
    duals = np.array([model.getRowDualSol(row) for row in rows]) / (row_norms * obj_norm)
    row_feats['dual'] = np.concatenate((-duals[has_lhs], duals[has_rhs])).reshape(-1, 1)

    # 合并特征
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

    # 稀疏矩阵
    coef_matrix = sp.csr_matrix((data[:, 0], (data[:, 1], data[:, 2])), shape=(n_rows, n_cols))
    coef_matrix = sp.vstack((-coef_matrix[has_lhs, :], coef_matrix[has_rhs, :])).tocoo(copy=False)

    row_ind, col_ind = coef_matrix.row, coef_matrix.col
    edge_feats = {'coef': coef_matrix.data.reshape(-1, 1)}

    cons_edge_feats = {
        'features': ['coef'],
        'indices': np.vstack([row_ind, col_ind]),
        'values': coef_matrix.data.reshape(-1, 1)
    }

    #=============== 变量（列）特征 ===============
    col_feats = {}

    # 类型
    type_map = {'BINARY': 0, 'INTEGER': 1, 'IMPLINT': 2, 'CONTINUOUS': 3}
    types = np.array([type_map[col.getVar().vtype()] for col in cols])
    col_feats['type'] = np.zeros((n_cols, 4))
    col_feats['type'][np.arange(n_cols), types] = 1

    # 目标系数
    col_feats['obj_coef'] = np.array([col.getObjCoeff() for col in cols]).reshape(-1, 1) / obj_norm

    # 边界
    lb = np.array([col.getLb() for col in cols])
    ub = np.array([col.getUb() for col in cols])
    has_lb = [not model.isInfinity(-val) for val in lb]
    has_ub = [not model.isInfinity(val) for val in ub]
    col_feats['has_lb'] = np.array(has_lb).astype(int).reshape(-1, 1)
    col_feats['has_ub'] = np.array(has_ub).astype(int).reshape(-1, 1)
    col_feats['at_lb'] = np.array([col.getBasisStatus() == 'lower' for col in cols]).reshape(-1, 1)
    col_feats['at_ub'] = np.array([col.getBasisStatus() == 'upper' for col in cols]).reshape(-1, 1)

    # 分数度
    col_feats['frac'] = np.array(
        [0.5 - abs(col.getVar().getLPSol() - floor(col.getVar().getLPSol()) - 0.5) for col in cols]
    ).reshape(-1, 1)
    col_feats['frac'][types == 3] = 0

    # 约化代价
    col_feats['reduced_cost'] = np.array([model.getVarRedcost(col.getVar()) for col in cols]).reshape(-1, 1) / obj_norm

    # LP解
    col_feats['lp_val'] = np.array([col.getVar().getLPSol() for col in cols]).reshape(-1, 1)

    # 原始解
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

    # 合并
    col_feat_names = []
    for k, v in col_feats.items():
        if v.shape[1] == 1:
            col_feat_names.append(k)
        else:
            col_feat_names.extend([f'{k}_{i}' for i in range(v.shape[1])])

    col_feat_vals = np.concatenate(list(col_feats.values()), axis=-1)
    var_feats = {'features': col_feat_names, 'values': col_feat_vals}

    #=============== 割平面特征 ===============
    cut_feats = {}

    # 范数
    cut_norms = np.array([cut.getNorm() for cut in cuts])
    cut_norms[cut_norms == 0] = 1

    # RHS（选择最违反的一侧）
    activity = np.array([model.getRowActivity(cut) for cut in cuts])
    lhs_cut = np.array([cut.getLhs() for cut in cuts])
    rhs_cut = np.array([cut.getRhs() for cut in cuts])
    has_lhs_cut = [not model.isInfinity(-val) for val in lhs_cut]
    has_lhs_cut = np.logical_and(has_lhs_cut, (lhs_cut - activity) > (activity - rhs_cut))
    has_rhs_cut = np.logical_not(has_lhs_cut)

    cut_feats['rhs'] = np.concatenate(
        (-(lhs_cut / cut_norms)[has_lhs_cut], (rhs_cut / cut_norms)[has_rhs_cut])
    ).reshape(-1, 1)

    # 支撑度
    cut_feats['support'] = np.array([len(cut.getCols()) / n_cols for cut in cuts]).reshape(-1, 1)

    # 违反度
    violations = np.concatenate(((lhs_cut - activity)[has_lhs_cut], (activity - rhs_cut)[has_rhs_cut]))
    cut_feats['violation'] = (violations / cut_norms).reshape(-1, 1)

    # 合并
    cut_feat_names = []
    for k, v in cut_feats.items():
        if v.shape[1] == 1:
            cut_feat_names.append(k)
        else:
            cut_feat_names.extend([f'{k}_{i}' for i in range(v.shape[1])])

    cut_feat_vals = np.concatenate(list(cut_feats.values()), axis=-1)
    cut_feats_dict = {'features': cut_feat_names, 'values': cut_feat_vals}

    #=============== 割平面-变量边特征 ===============
    cut_data = []
    for i, cut in enumerate(cuts):
        cols_in_cut = cut.getCols()
        vals_in_cut = cut.getVals()
        for j, (col, val) in enumerate(zip(cols_in_cut, vals_in_cut)):
            cut_data.append([val / cut_norms[i if has_rhs_cut[i] else i],
                           i, col.getLPPos()])

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

    # 转换为所需格式
    state = {
        'constraint_features': cons_feats['values'],
        'constraint_feat_names': cons_feats['features'],

        'variable_features': var_feats['values'],
        'variable_feat_names': var_feats['features'],

        'cut_features': cut_feats_dict['values'],
        'cut_feat_names': cut_feats_dict['features'],

        'var_cons_edges': cons_edge_feats['indices'],
        'var_cons_edge_features': cons_edge_feats['values'],
        'var_cons_edge_feat_names': cons_edge_feats['features'],

        'var_cut_edges': cut_edge_feats['indices'],
        'var_cut_edge_features': cut_edge_feats['values'],
        'var_cut_edge_feat_names': cut_edge_feats['features'],

        'n_vars': var_feats['values'].shape[0],
        'n_cons': cons_feats['values'].shape[0],
        'n_cuts': cut_feats_dict['values'].shape[0]
    }

    return state
