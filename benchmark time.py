# benchmark_ctng.py
# 用途：测量 CTNG 模块在不同规模数据集上的运行时间
# 计时范围：
#   - T_knn  : build_knn_index（KNN索引构建，只建一次）
#   - T_ctng : search_TN + divide + clustering + smart_merge（单次k扫描）
#   - T_total: 完整 drecord 扫描（遍历全部k值）的总耗时
# 运行方式：python benchmark_ctng.py
# 输出：控制台打印 + result/benchmark_timing.csv

import os
import time
import numpy as np
import pandas as pd

from search_TN_optimized import build_knn_index, search_TN_optimized
from divide      import divide
from clustering  import clustering
from improvements import compute_adaptive_beta, smart_merge_small_clusters

# ─────────────────────────────────────────────────────────────────────────────
# 配置区：修改这里
# ─────────────────────────────────────────────────────────────────────────────

# 每个数据集对应的 latent 文件路径和细胞数（已有的潜在表示，直接加载）
DATASETS = [
    {"name": "Trachea",           "N": 489,   "latent": "DimensionalityReduction/Trachea/Trachea_latent_zinb.csv"},
    {"name": "EyeRetina",         "N": 1062,  "latent": "DimensionalityReduction/EyeRetina/EyeRetina_latent_zinb.csv"},
    {"name": "Uterus",            "N": 1685,  "latent": "DimensionalityReduction/Uterus/Uterus_latent_zinb.csv"},
    {"name": "Baron_Mouse",       "N": 1886,  "latent": "DimensionalityReduction/Baron_Mouse/Baron_Mouse_latent_zinb.csv"},
    {"name": "Muraro",            "N": 2122,  "latent": "DimensionalityReduction/Muraro/Muraro_latent_zinb.csv"},
    {"name": "Segerstolpe",       "N": 2133,  "latent": "DimensionalityReduction/Segerstolpe/Segerstolpe_latent_zinb.csv"},
    {"name": "Baron_Human",       "N": 8569,  "latent": "DimensionalityReduction/Baron_Human/Baron_Human_latent_zinb.csv"},
    {"name": "Blood",             "N": 9354,  "latent": "DimensionalityReduction/Blood_new/Blood_new_latent_zinb.csv"},
    {"name": "Immune_Health",     "N": 10937, "latent": "DimensionalityReduction/Immune_Health/Immune_Health_latent_zinb.csv"},
    {"name": "Immune_Health_DCs", "N": 23287, "latent": "DimensionalityReduction/Immune_Health_DCs/Immune_Health_DCs_latent_zinb.csv"},
]

K_FIXED      = 30      # 固定k值，所有数据集统一，消除k不同带来的干扰
BASE_BETA    = 0.3     # 固定beta，不用自适应（避免compute_adaptive_beta的时间干扰）
N_REPEAT     = 5       # 重复次数，取均值和标准差
RANDOM_STATE = 3407

# ─────────────────────────────────────────────────────────────────────────────
# 单次 CTNG 计时（不含 KNN 构建）
# ─────────────────────────────────────────────────────────────────────────────
def time_ctng_single(X_latent, k, knn_indices, knn_distances,
                     base_beta=0.3):
    """
    对已有的 knn_indices 跑一次完整 CTNG，返回耗时（秒）。
    包含：search_TN + divide + clustering + smart_merge
    不包含：build_knn_index（单独计时）
    """
    ki_tn = knn_indices[:, :k + 1]
    kd_tn = knn_distances[:, :k + 1]

    t0 = time.perf_counter()

    alpha, TN, ki_tn, kd_tn = search_TN_optimized(
        X_latent, k,
        knn_indices=ki_tn,
        knn_distances=kd_tn,
        random_state=RANDOM_STATE,
    )
    cl_point, noise = divide(X_latent, TN, ki_tn, k, alpha)
    Clusters, cl_number = clustering(
        noise, cl_point, X_latent, knn_indices, k, TN, base_beta
    )
    Clusters = smart_merge_small_clusters(
        Clusters, X_latent, min_size_ratio=0.001
    )

    elapsed = time.perf_counter() - t0
    return elapsed, cl_number


# ─────────────────────────────────────────────────────────────────────────────
# 主循环
# ─────────────────────────────────────────────────────────────────────────────
def main():
    np.random.seed(RANDOM_STATE)
    os.makedirs("result", exist_ok=True)

    rows = []

    print(f"{'='*70}")
    print(f"CTNG 计时基准  (k={K_FIXED}, repeat={N_REPEAT})")
    print(f"{'='*70}")
    print(f"{'Dataset':<22} {'N':>7}  {'T_knn(s)':>10}  "
          f"{'T_ctng_mean(s)':>15}  {'T_ctng_std(s)':>14}  {'Clusters':>9}")
    print(f"{'-'*70}")

    for ds in DATASETS:
        name   = ds["name"]
        N_true = ds["N"]
        path   = ds["latent"]

        # ── 加载潜在表示 ──────────────────────────────────────────────────
        if not os.path.exists(path):
            print(f"[跳过] {name}: 文件不存在 → {path}")
            continue

        X_latent = pd.read_csv(path, index_col=0).values.astype("float32")
        N_actual = X_latent.shape[0]

        # ── 计时 KNN 构建（只建一次，不重复）────────────────────────────
        t_knn_start = time.perf_counter()
        knn_indices, knn_distances = build_knn_index(
            X_latent, K_FIXED * 2, random_state=RANDOM_STATE
        )
        t_knn = time.perf_counter() - t_knn_start

        # ── 计时单次 CTNG（重复 N_REPEAT 次）────────────────────────────
        # 第一次可能有缓存预热，丢弃第一次
        times = []
        cl_number = 0
        for i in range(N_REPEAT + 1):
            t, cl = time_ctng_single(
                X_latent, K_FIXED,
                knn_indices, knn_distances,
                base_beta=BASE_BETA,
            )
            if i > 0:       # 跳过第一次（预热）
                times.append(t)
                cl_number = cl

        t_mean = np.mean(times)
        t_std  = np.std(times)

        print(f"{name:<22} {N_actual:>7}  {t_knn:>10.3f}  "
              f"{t_mean:>15.4f}  {t_std:>14.4f}  {cl_number:>9}")

        rows.append({
            "dataset":       name,
            "N":             N_actual,
            "T_knn_s":       round(t_knn, 4),
            "T_ctng_mean_s": round(t_mean, 4),
            "T_ctng_std_s":  round(t_std, 4),
            "n_clusters":    cl_number,
            "k_fixed":       K_FIXED,
        })

    print(f"{'='*70}")

    # ── 保存结果 ─────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    out_path = "result/benchmark_timing.csv"
    df.to_csv(out_path, index=False)
    print(f"\n结果已保存至 {out_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()