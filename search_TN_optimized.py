# search_TN_optimized.py
import numpy as np
from pynndescent import NNDescent

# ─────────────────────────────────────────────────────────────────────────────
# 核心改动：用 pynndescent 近似 KNN 替代 squareform(pdist) 精确距离矩阵
#
# 原代码问题
# ──────────
# squareform(pdist(X)) 需要构造完整的 N×N 距离矩阵。
# 以 HumanCellAtlas 约 10 万细胞为例：
#   10^5 × 10^5 × 4 bytes (float32) ≈ 37 GB
# 这在任何普通工作站上都会 OOM。
# 即使数据集只有 2 万细胞，也需要约 1.5 GB，且计算时间是 O(N²·d)。
#
# 解决方案：近似 K 近邻（Approximate Nearest Neighbor, ANN）
# ────────────────────────────────────────────────────────
# pynndescent 实现了 NN-Descent 算法（Dong et al., 2011），
# 复杂度约为 O(N · k · log N)，内存仅需存储 N×(k+1) 的稀疏邻居矩阵。
# 对单细胞数据集（32 维潜在空间），近似误差极小（召回率 >95%）。
#
# 数据结构变化
# ────────────
# 原代码维护了：
#   A     : N×N float 距离矩阵（稠密）
#   Xu    : N×N int   排序索引（稠密）
#   XIndex: list of list（长度 N，每个子列表长度 N）
#
# 新代码只维护：
#   knn_indices  : N×(k+1) int   —— 第 i 行存点 i 的 k+1 个最近邻索引（含自身）
#   knn_distances: N×(k+1) float —— 对应距离（可选，噪声分配时用到）
# ─────────────────────────────────────────────────────────────────────────────


def build_knn_index(X, k, n_jobs=-1, random_state=3407):
    """
    用 pynndescent 构建近似 KNN 索引。

    参数
    ────
    X            : (N, d) 数据矩阵，float32
    k            : 邻居数量（不含自身）
    n_jobs       : 并行线程数，-1 表示使用全部 CPU 核
    random_state : 随机种子，保证可复现

    返回
    ────
    knn_indices   : (N, k+1) int   —— 每行第 0 列是点自身
    knn_distances : (N, k+1) float
    """
    # NNDescent 要求查询时返回 k+1 个邻居（第 0 个是点自身，距离=0）
    # k*2+1：前 k+1 列用于 TN 计算，额外 k 列专门保证噪声分配时
    # 能扫描到足够多的非噪声邻居（噪声点多时 k+1 列可能全是噪声）
    index = NNDescent(
        X,
        n_neighbors=k * 2 + 1,
        metric='euclidean',
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=False
    )
    # neighbor_graph 返回 (indices, distances)，形状均为 (N, k+1)
    knn_indices, knn_distances = index.neighbor_graph
    return knn_indices, knn_distances


def search_TN_optimized(X, k,
                        knn_indices=None, knn_distances=None,
                        n_jobs=-1, random_state=3407):
    """
    基于近似 KNN 计算紧邻集（Tight Neighborhood, TN）。

    紧邻定义：TN(i, r) = KNN_r(i) ∩ RKNN_r(i)
    即：点 i 的第 r 层紧邻 = 既是 i 的第 r 近邻、又把 i 列为其第 r 近邻的点集。

    参数
    ────
    X              : (N, d) 数据矩阵
    k              : 最大紧邻层数
    knn_indices    : 预计算的 KNN 索引 (N, k+1)，None 则自动构建
    knn_distances  : 预计算的 KNN 距离 (N, k+1)，None 则自动构建
    n_jobs         : 并行线程数
    random_state   : 随机种子

    返回
    ────
    alpha        : 核心点阈值（mean - std of TN_k 大小）
    TN           : list[list[list]]，TN[i][r] = 点 i 第 r 层紧邻的索引列表
    knn_indices  : (N, k+1) int，供 divide / clustering 复用（替代原 Xu/XIndex）
    knn_distances: (N, k+1) float
    """
    N = X.shape[0]

    # ── 1. 构建 KNN 索引（若未预计算）────────────────────────────────────
    if knn_indices is None or knn_distances is None:
        print(f"  构建近似 KNN 索引 (N={N}, k={k})...")
        knn_indices, knn_distances = build_knn_index(
            X, k, n_jobs=n_jobs, random_state=random_state
        )

    # ── 2. 构建 KNN / RKNN 集合（set 化加速交集运算）────────────────────
    # knn_indices[:, 0] 是点自身，邻居从第 1 列开始
    # KNN_sets[i] = 点 i 的 k 个最近邻（set，不含自身）
    KNN_sets = [set(knn_indices[i, 1:k + 1].tolist()) for i in range(N)]

    # RKNN_sets[i] = 把 i 列为邻居的点（反向 KNN）
    RKNN_sets = [set() for _ in range(N)]
    for i in range(N):
        for j in knn_indices[i, 1:k + 1]:
            RKNN_sets[j].add(i)

    # ── 3. 逐层计算 TN ───────────────────────────────────────────────────
    # TN[i][r] = 第 i 点的第 (r+1)-紧邻集合（r 从 0 开始，对应原代码 1-indexed）
    #
    # 原代码的 KNN/RKNN 是累积的（第 r 层包含前 r 层邻居），
    # 导致高层 TN 集合越来越大、计算交集代价递增。
    # 这里维持相同语义：第 r 层 KNN = 前 r+1 个邻居（不含自身）。
    TN = [[[] for _ in range(k)] for _ in range(N)]

    # 用增量方式：每层只新增一个邻居，累积 KNN_acc / RKNN_acc
    KNN_acc  = [set() for _ in range(N)]
    RKNN_acc = [set() for _ in range(N)]

    for r in range(k):
        # 新增第 r+1 个邻居（knn_indices 第 r+1 列，0 列是自身）
        for i in range(N):
            new_nb = int(knn_indices[i, r + 1])
            KNN_acc[i].add(new_nb)
            RKNN_acc[new_nb].add(i)

        # TN[i][r] = 截至第 r 层的 KNN ∩ RKNN
        for i in range(N):
            TN[i][r] = list(KNN_acc[i] & RKNN_acc[i])

    # ── 4. 计算阈值 alpha ────────────────────────────────────────────────
    TN_num  = np.array([len(TN[i][k - 1]) for i in range(N)], dtype=float)
    TN_mean = TN_num.mean()
    TN_std  = TN_num.std()
    alpha   = max(0.0, TN_mean - TN_std)

    return alpha, TN, knn_indices, knn_distances


if __name__ == "__main__":
    print("search_TN_optimized 模块已加载")