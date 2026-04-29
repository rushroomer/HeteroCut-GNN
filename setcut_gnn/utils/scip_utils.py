"""
SCIP 工具函数
"""

from pyscipopt import Model


def init_scip(model: Model, seed: int, time_limit: int = 300):
    """初始化 SCIP 求解器参数

    Args:
        model: SCIP 模型
        seed: 随机种子
        time_limit: 时间限制（秒）
    """
    # 确保seed在合理范围内 (SCIP的int参数范围)
    seed = int(seed) % 100000

    # 基础设置
    model.setIntParam('display/verblevel', 0)
    model.setRealParam('limits/time', time_limit)

    # 随机化设置
    try:
        model.setBoolParam('randomization/permutevars', True)
        model.setIntParam('randomization/permutationseed', seed)
        model.setIntParam('randomization/randomseedshift', seed)
    except KeyError:
        # 如果参数不存在，使用其他方式设置种子
        pass

    # 割平面设置
    try:
        model.setIntParam('separating/maxrounds', 10)
        model.setIntParam('separating/maxroundsroot', 10)
        model.setIntParam('separating/maxcuts', 50)
        model.setIntParam('separating/maxcutsroot', 50)
    except KeyError:
        pass

    # 禁用预处理（保持问题原始结构）
    try:
        model.setIntParam('presolving/maxrounds', 0)
        model.setIntParam('presolving/maxrestarts', 0)
    except KeyError:
        pass


def get_lp_bound(model: Model) -> float:
    """获取当前 LP 松弛下界"""
    return model.getLPObjVal()


def get_primal_bound(model: Model) -> float:
    """获取当前原始上界"""
    return model.getPrimalbound()
