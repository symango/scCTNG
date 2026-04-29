# clustering.py
import numpy as np


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank   = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1

    def same(self, x, y):
        return self.find(x) == self.find(y)

    def components(self, nodes=None):
        from collections import defaultdict
        groups = defaultdict(list)
        iterable = nodes if nodes is not None else range(len(self.parent))
        for i in iterable:
            groups[self.find(i)].append(i)
        return dict(groups)


def clustering(noise, cl_point, X, knn_indices, k, TN, beta):
    """
    CTNG 聚类。

    噪声分配说明
    ────────────
    恢复原始的"找到第一个非噪声邻居就合并"逻辑，移除距离门槛。
    距离门槛会导致大量噪声点找不到合并目标，每个孤立噪声点自成一簇，
    造成类数居高不下、k 增大时类数下降不明显的问题。

    噪声点找不到非噪声邻居的根本原因是 knn_indices 列数不足（只有 k+1 列）。
    当噪声点数量多时，它们互相成为彼此的近邻，k+1 列全是噪声点，
    导致扫描完所有列都找不到非噪声邻居。

    解决方式：build_knn_index 构建时用 k*2 列（在 search_TN_optimized
    和 drecord 里统一处理），保证噪声分配时有足够多的列可以扫描。
    clustering 本身恢复简单可靠的原始逻辑。
    """
    N = X.shape[0]

    noise_set = set(noise)

    TN_sets = [
        [frozenset(TN[i][r]) for r in range(k)]
        for i in range(N)
    ]

    uf = UnionFind(N)

    # ── 第一步：2-紧邻直接连接 ────────────────────────────────────────────
    for ci in cl_point:
        tn2 = TN_sets[ci][1] - noise_set
        for nb in tn2:
            if nb > ci:
                uf.union(ci, nb)

    # ── 第二步：高层紧邻按 beta 连接 ─────────────────────────────────────
    for layer in range(2, k):
        comps = uf.components(nodes=cl_point)
        visited = set(noise_set)

        for root, comp_nodes in comps.items():
            visited.update(comp_nodes)

            for p in comp_nodes:
                tn_p = TN_sets[p][layer]
                candidates = tn_p - visited

                for q in list(candidates):
                    tn_q   = TN_sets[q][layer]
                    common = tn_p & tn_q
                    denom  = min(len(tn_p), len(tn_q))
                    if denom == 0:
                        continue
                    w = len(common) / denom

                    if w > beta:
                        uf.union(p, q)
                        visited.add(q)

    # ── 第三步：小聚类归为噪声 ────────────────────────────────────────────
    comps_all = uf.components()
    for root, members in comps_all.items():
        if len(members) <= 3:
            noise_set.update(members)

    # ── 第四步：噪声点分配 ────────────────────────────────────────────────
    # 扫描 knn_indices 所有列，找第一个非噪声邻居合并。
    # knn_indices 的列数由 build_knn_index 的 k 参数决定，
    # drecord 和 main 里构建时统一用 k*2，保证这里有足够列可扫描。
    for ni in list(noise_set):
        for col in range(1, knn_indices.shape[1]):
            nb = int(knn_indices[ni, col])
            if nb not in noise_set:
                uf.union(ni, nb)
                break

    # ── 生成最终标签 ──────────────────────────────────────────────────────
    comps_final = uf.components()
    cl_number   = len(comps_final)

    Clusters = np.zeros(N, dtype=int)
    for label_id, (root, members) in enumerate(comps_final.items(), start=1):
        for node in members:
            Clusters[node] = label_id

    return Clusters, cl_number