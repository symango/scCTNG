# benchmark_ctng_krange.py
# 用途：测量 CTNG 模块在不同规模数据集上遍历完整 k_range 的运行时间
#
# 计时分为三层：
#   T_knn    : build_knn_index 一次性构建（秒）
#   T_per_k  : 单个 k 值的 CTNG 耗时均值±std（秒）
#   T_total  : 遍历完整 k_range 的总耗时（秒）—— 这是论文里应报告的数字
#
# 运行：python benchmark_ctng_krange.py
# 输出：result/benchmark_krange.csv  +  result/benchmark_krange_plot.png

import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from search_TN_optimized import build_knn_index, search_TN_optimized
from divide      import divide
from clustering  import clustering
from improvements import compute_adaptive_beta, smart_merge_small_clusters

# ─────────────────────────────────────────────────────────────────────────────
# 配置区
# ─────────────────────────────────────────────────────────────────────────────

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

# k_range 与你主实验保持一致
K_RANGE      = range(3, 41)      # 从3到40，共38个k值
BASE_BETA    = 0.3
N_REPEAT     = 3                 # 整个 k_range 重复几次取均值（大数据集跑3次即可）
RANDOM_STATE = 3407

# ─────────────────────────────────────────────────────────────────────────────
# 单次完整 k_range 扫描（复用同一个 knn_indices）
# ─────────────────────────────────────────────────────────────────────────────
def run_full_krange(X_latent, k_range, knn_indices, knn_distances,
                    base_beta=0.3):
    """
    遍历 k_range 中每个 k，执行一次完整 CTNG，返回总耗时和每个 k 的耗时列表。
    knn_indices / knn_distances 已预先构建，不计入此函数的计时。
    """
    k_times = []
    t_total_start = time.perf_counter()

    for k in k_range:
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
        Clusters, _ = clustering(
            noise, cl_point, X_latent, knn_indices, k, TN, base_beta
        )
        smart_merge_small_clusters(Clusters, X_latent, min_size_ratio=0.001)
        k_times.append(time.perf_counter() - t0)

    t_total = time.perf_counter() - t_total_start
    return t_total, k_times


# ─────────────────────────────────────────────────────────────────────────────
# 主循环
# ─────────────────────────────────────────────────────────────────────────────
def main():
    np.random.seed(RANDOM_STATE)
    os.makedirs("result", exist_ok=True)

    k_max = max(K_RANGE)
    rows  = []

    print(f"{'='*72}")
    print(f"CTNG 计时基准  (k_range={min(K_RANGE)}~{k_max}, repeat={N_REPEAT})")
    print(f"{'='*72}")
    print(f"{'Dataset':<22} {'N':>7}  {'T_knn(s)':>9}  "
          f"{'T_per_k_mean(s)':>16}  {'T_total_mean(s)':>16}  {'T_total_std(s)':>14}")
    print(f"{'-'*72}")

    for ds in DATASETS:
        name   = ds["name"]
        path   = ds["latent"]

        if not os.path.exists(path):
            print(f"[跳过] {name}: 文件不存在 → {path}")
            continue

        X_latent = pd.read_csv(path, index_col=0).values.astype("float32")
        N_actual = X_latent.shape[0]

        # ── 计时 KNN 构建（只建一次）────────────────────────────────────
        t_knn_start = time.perf_counter()
        knn_indices, knn_distances = build_knn_index(
            X_latent, k_max * 2, random_state=RANDOM_STATE
        )
        t_knn = time.perf_counter() - t_knn_start

        # ── 计时完整 k_range 扫描（重复 N_REPEAT 次）────────────────────
        # 第一次预热，丢弃
        total_times = []
        all_k_times = []
        for i in range(N_REPEAT + 1):
            t_total, k_times = run_full_krange(
                X_latent, K_RANGE,
                knn_indices, knn_distances,
                base_beta=BASE_BETA,
            )
            if i > 0:
                total_times.append(t_total)
                all_k_times.append(k_times)

        t_total_mean = np.mean(total_times)
        t_total_std  = np.std(total_times)

        # 每个 k 值的平均耗时（用于分析哪个 k 最慢）
        per_k_mean = np.mean(all_k_times, axis=0)
        t_per_k_mean = float(np.mean(per_k_mean))

        print(f"{name:<22} {N_actual:>7}  {t_knn:>9.3f}  "
              f"{t_per_k_mean:>16.4f}  {t_total_mean:>16.3f}  {t_total_std:>14.4f}")

        rows.append({
            "dataset":         name,
            "N":               N_actual,
            "k_range":         f"{min(K_RANGE)}-{k_max}",
            "T_knn_s":         round(t_knn, 4),
            "T_per_k_mean_s":  round(t_per_k_mean, 4),
            "T_total_mean_s":  round(t_total_mean, 3),
            "T_total_std_s":   round(t_total_std, 4),
        })

    print(f"{'='*72}")

    # ── 保存 CSV ─────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    csv_path = "result/benchmark_krange.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n结果已保存至 {csv_path}")
    print(df.to_string(index=False))

    # ── 绘图（T_total vs N，论文直接用这张图）────────────────────────────
    if len(rows) < 2:
        print("数据点不足，跳过绘图")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    Ns      = [r["N"] for r in rows]
    totals  = [r["T_total_mean_s"] for r in rows]
    stds    = [r["T_total_std_s"] for r in rows]
    names   = [r["dataset"] for r in rows]
    t_knns  = [r["T_knn_s"] for r in rows]

    # 左图：总耗时 vs N（含误差棒）
    ax = axes[0]
    ax.errorbar(Ns, totals, yerr=stds, fmt='o-', color='#534AB7',
                capsize=4, linewidth=1.5, markersize=5,
                label=f'Full k-scan ({min(K_RANGE)}~{k_max})')
    ax.bar(Ns, t_knns, width=[n*0.04 for n in Ns],
           alpha=0.3, color='#1D9E75', label='KNN index build')
    for i, (n, t, name) in enumerate(zip(Ns, totals, names)):
        ax.annotate(name, (n, t), textcoords="offset points",
                    xytext=(4, 4), fontsize=7, color='#444441')
    ax.set_xlabel('Number of cells (N)', fontsize=11)
    ax.set_ylabel('Runtime (seconds)', fontsize=11)
    ax.set_title(f'CTNG runtime vs. dataset scale\n(k = {min(K_RANGE)} to {k_max}, CPU single-thread)',
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 右图：log-log scale（便于观察复杂度增长趋势）
    ax2 = axes[1]
    ax2.loglog(Ns, totals, 'o-', color='#534AB7', linewidth=1.5, markersize=5)
    # 拟合参考线（O(N^2)）
    N_arr = np.array(Ns, dtype=float)
    ref_scale = totals[0] / (Ns[0] ** 2)
    ref_line  = ref_scale * N_arr ** 2
    ax2.loglog(Ns, ref_line, '--', color='#E24B4A', alpha=0.6, label='O(N²) reference')
    ax2.set_xlabel('Number of cells (N)', fontsize=11)
    ax2.set_ylabel('Runtime (seconds)', fontsize=11)
    ax2.set_title('Log-log scale (complexity trend)', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plot_path = "result/benchmark_krange_plot.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"图表已保存至 {plot_path}")


if __name__ == "__main__":
    main()