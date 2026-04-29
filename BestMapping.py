# BestMapping.py
import numpy as np
from scipy.optimize import linear_sum_assignment


def BestMapping(true_labels, pred_labels):
    """
    使用匈牙利算法将预测标签映射到真实标签空间，最大化重叠数量。

    修复说明
    ────────
    原代码在 pred 类别数多于 true 类别数时存在 bug：
        new_labels[idx] = row_ind[col_ind == i][0] + 1 if ... else i + 1

    当 col_ind == i 找不到对应行时，映射逻辑不一致（混用 0-indexed 和 1-indexed），
    导致多余的预测类别全部映射到同一个标签，造成 ACC 虚高。

    修复方式
    ────────
    1. 统一构造 (max_classes × max_classes) 的方形共现矩阵，
       行对应 true 类别，列对应 pred 类别，超出部分补 0。
    2. 匈牙利算法在方形矩阵上求解全局最优映射（行→列）。
    3. 用 pred_class → mapped_true_class 的字典做批量替换，
       逻辑清晰且不依赖索引运算的顺序。

    参数
    ────
    true_labels : array-like，真实标签
    pred_labels : array-like，预测标签

    返回
    ────
    new_labels : np.ndarray，映射后的预测标签（与 true_labels 同值域）
    """
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)

    true_classes = np.unique(true_labels)
    pred_classes = np.unique(pred_labels)

    n_true = len(true_classes)
    n_pred = len(pred_classes)
    n_max  = max(n_true, n_pred)

    # ── 构建方形共现矩阵 ──────────────────────────────────────────────────
    # G[i, j] = 真实类别 i 且预测类别 j 的样本数
    G = np.zeros((n_max, n_max), dtype=np.int64)
    for i, tc in enumerate(true_classes):
        true_mask = (true_labels == tc)
        for j, pc in enumerate(pred_classes):
            G[i, j] = np.sum(true_mask & (pred_labels == pc))

    # ── 匈牙利算法（最大化 → 取负值最小化）──────────────────────────────
    row_ind, col_ind = linear_sum_assignment(-G)
    # row_ind[t] = true 类别索引，col_ind[t] = 被映射到该 true 类别的 pred 类别索引

    # ── 构建 pred_class → true_class 映射字典 ────────────────────────────
    mapping = {}
    for r, c in zip(row_ind, col_ind):
        if c < n_pred:          # 只映射实际存在的预测类别
            mapping[pred_classes[c]] = true_classes[r] if r < n_true else r + 1

    # 若某些预测类别未被匹配（n_pred > n_true 时剩余的列），赋予唯一标签
    max_true = int(true_classes.max()) if n_true > 0 else 0
    extra_id = max_true + 1
    for pc in pred_classes:
        if pc not in mapping:
            mapping[pc] = extra_id
            extra_id += 1

    # ── 批量替换 ──────────────────────────────────────────────────────────
    new_labels = np.vectorize(mapping.get)(pred_labels)

    return new_labels
