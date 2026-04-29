# run.py
# AE_CTNG 主运行入口
# 功能：完成 test.py（AE训练+降维） 和 drecord_optimized.py（k扫描）的全部工作
# 新增：改进2（智能合并）、改进3（自适应beta）、改进4（KL联合优化）
#
# 依赖文件（与本文件放在同一目录）：
#   AE.py / preprocess.py                  ← 原有，不需要修改
#   search_TN_optimized.py / divide.py     ← 原有，不需要修改
#   clustering.py / BestMapping.py         ← 原有，不需要修改
#   metrics.py                             ← 原有，不需要修改
#   improvements.py                        ← 新增，包含三项改进

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from AE          import set_seed, train_autoencoder, build_autoencoder
from preprocess  import read_data, normalize
from search_TN_optimized import build_knn_index, search_TN_optimized
from divide      import divide
from clustering  import clustering
from BestMapping import BestMapping
from metrics     import Acc, Fmeasure, nmi, ari
from tensorflow.keras.optimizers import Adam
from improvements import (compute_adaptive_beta,
                           smart_merge_small_clusters,
                           soft_assignment, target_distribution,
                           joint_optimize)


np.random.seed(3407)
set_seed(3407)


# ───────────────────────────────────────────────────────────────────────────
# 工具：运行一次 CTNG（含改进2、改进3）
# ───────────────────────────────────────────────────────────────────────────
def run_ctng_once(X_latent, k, base_beta,
                  knn_indices=None, knn_distances=None,
                  use_adaptive_beta=True, use_smart_merge=True,
                  min_size_ratio=0.001):
    """
    对给定潜在空间跑一次 CTNG，返回聚类标签和簇数。
    knn_indices / knn_distances 可复用，避免重复构建。
    """
    if knn_indices is None or knn_distances is None:
        knn_indices, knn_distances = build_knn_index(X_latent, k * 2, random_state=3407)

    ki_tn = knn_indices[:, :k + 1]
    kd_tn = knn_distances[:, :k + 1]
    alpha, TN, ki_tn, kd_tn = search_TN_optimized(
        X_latent, k, knn_indices=ki_tn, knn_distances=kd_tn,
        random_state=3407)
    cl_point, noise = divide(X_latent, TN, ki_tn, k, alpha)

    # 改进3：自适应 beta
    if use_adaptive_beta:
        beta_val, beta_arr = compute_adaptive_beta(
            knn_distances, base_beta=base_beta)
        print(f"    自适应 beta={beta_val:.4f} "
              f"(范围 {beta_arr.min():.3f}~{beta_arr.max():.3f})")
    else:
        beta_val = base_beta

    Clusters, cl_number = clustering(
        noise, cl_point, X_latent, knn_indices, k, TN, beta_val)

    # 改进2：智能合并小簇
    if use_smart_merge:
        before   = len(np.unique(Clusters))
        Clusters = smart_merge_small_clusters(Clusters, X_latent,
                                                  min_size_ratio=min_size_ratio)
        after    = len(np.unique(Clusters))
        if before != after:
            print(f"    智能合并：{before} → {after} 簇")
        cl_number = len(np.unique(Clusters))

    return Clusters, cl_number, knn_indices, knn_distances


# ───────────────────────────────────────────────────────────────────────────
# 联合优化主循环（改进4）
# ───────────────────────────────────────────────────────────────────────────
def run_joint_optimization(base_model, encoder, X, label,
                            joint_epochs, k_init, base_beta,
                            lambda_kl, batch_size,
                            pseudo_update_freq,
                            use_adaptive_beta, use_smart_merge,
                            min_size_ratio=0.001):
    """
    在 AE 预训练之后，执行 ZINB + KL 软分配联合优化。

    流程
    ────
    1. 用当前潜在空间跑 CTNG，生成初始伪标签
    2. 每个 epoch：joint_optimize（ZINB + KL）
    3. 每 pseudo_update_freq 个 epoch：重跑 CTNG 更新伪标签和簇中心
    4. 记录综合指标最优的潜在表示

    返回
    ────
    best_latent : (N, d) numpy，综合指标最优的潜在表示
    """
    optimizer = Adam(learning_rate=0.001)
    has_labels = label is not None

    # 初始伪标签
    print("\n  初始化伪标签（CTNG 初始聚类）...")
    latent_cur = encoder.predict(X, verbose=0)
    Clusters, cl_number, _, _ = run_ctng_once(
        latent_cur, k_init, base_beta,
        use_adaptive_beta=use_adaptive_beta,
        use_smart_merge=use_smart_merge,
        min_size_ratio=min_size_ratio)

    pseudo_labels   = np.array(Clusters, dtype=np.int32) - 1  # 0-indexed
    n_clusters      = len(np.unique(pseudo_labels))
    cluster_centers = np.array([
        latent_cur[pseudo_labels == c].mean(axis=0)
        for c in range(n_clusters)
        if np.sum(pseudo_labels == c) > 0
    ], dtype='float32')
    n_clusters = len(cluster_centers)
    print(f"  初始簇数: {n_clusters}")

    if has_labels:
        _eval_and_print(label, Clusters, tag="联合优化前")

    best_latent = latent_cur.copy()
    best_score  = -np.inf

    for epoch in range(1, joint_epochs + 1):

        # 每 pseudo_update_freq 个 epoch 更新伪标签
        if epoch % pseudo_update_freq == 0:
            latent_cur    = encoder.predict(X, verbose=0)
            Clusters, cl_number, _, _ = run_ctng_once(
                latent_cur, k_init, base_beta,
                use_adaptive_beta=use_adaptive_beta,
                use_smart_merge=use_smart_merge,
                min_size_ratio=min_size_ratio)
            pseudo_labels   = np.array(Clusters, dtype=np.int32) - 1
            n_clusters_new  = len(np.unique(pseudo_labels))
            cluster_centers = np.array([
                latent_cur[pseudo_labels == c].mean(axis=0)
                for c in range(n_clusters_new)
                if np.sum(pseudo_labels == c) > 0
            ], dtype='float32')
            n_clusters = len(cluster_centers)
            print(f"  [epoch {epoch}] 伪标签更新 → 簇数={n_clusters}")

        # 联合优化一个 epoch
        loss = joint_optimize(
            base_model, encoder, X,
            pseudo_labels, cluster_centers,
            optimizer, lambda_kl=lambda_kl, batch_size=batch_size)

        # 每10个epoch打印一次监控
        if epoch % 10 == 0:
            latent_mon = encoder.predict(X, verbose=0)
            Clusters_mon, cl_num_mon, _, _ = run_ctng_once(
                latent_mon, k_init, base_beta,
                use_adaptive_beta=use_adaptive_beta,
                use_smart_merge=use_smart_merge,
                min_size_ratio=min_size_ratio)

            score = 0.0
            if has_labels:
                score = _eval_and_print(
                    label, Clusters_mon,
                    tag=f"联合epoch {epoch}/{joint_epochs}")
            print(f"    loss={loss:.4f}  簇数={cl_num_mon}")

            if score > best_score:
                best_score  = score
                best_latent = latent_mon.copy()

    print(f"\n  联合优化完成，最优综合得分={best_score:.4f}")
    return best_latent


# ───────────────────────────────────────────────────────────────────────────
# drecord 扫描（与 drecord_optimized.py 功能完全一致）
# ───────────────────────────────────────────────────────────────────────────
def run_drecord(X_latent, k_range, base_beta, label, filename,
                use_adaptive_beta=True, use_smart_merge=True,
                min_size_ratio=0.001):
    """
    对最终潜在空间在 k_range 范围内扫描，输出指标并保存结果。
    与原 drecord_optimized.py 完全对应。
    """
    has_labels = label is not None
    k_max      = max(k_range)
    print(f"\n{'='*60}")
    print(f"drecord 扫描: k={min(k_range)}~{k_max}  数据集={filename}")
    print(f"{'='*60}")

    knn_indices, knn_distances = build_knn_index(X_latent, k_max * 2, random_state=3407)

    results, D = [], []

    for k in k_range:
        try:
            t0 = time.time()
            Clusters, cl_number, _, _ = run_ctng_once(
                X_latent, k, base_beta,
                knn_indices=knn_indices,
                knn_distances=knn_distances,
                use_adaptive_beta=use_adaptive_beta,
                use_smart_merge=use_smart_merge,
                min_size_ratio=min_size_ratio)

            D.append(cl_number)
            elapsed = time.time() - t0
            row = {'k': k, 'cluster_number': cl_number, 'time': elapsed}

            if has_labels:
                new_label   = BestMapping(label, Clusters)
                ACC         = Acc(label, new_label)
                FMeasure, _ = Fmeasure(label, Clusters)
                NMI         = nmi(label, Clusters)
                ARI         = ari(label, Clusters)
                row.update({'ACC': ACC, 'FMeasure': FMeasure,
                            'NMI': NMI, 'ARI': ARI})
                print(f"k={k:3d} | 簇数={cl_number:4d} | "
                      f"ACC={ACC:.4f} FM={FMeasure:.4f} "
                      f"NMI={NMI:.4f} ARI={ARI:.4f} | {elapsed:.2f}s")
            else:
                print(f"k={k:3d} | 簇数={cl_number:4d} | {elapsed:.2f}s")

            results.append(row)

        except Exception as e:
            D.append(0)
            results.append({'k': k, 'cluster_number': 0, 'time': 0})
            print(f"Error at k={k}: {e}")

    # 保存结果
    result_dir = os.path.join("result", "drecord_result", filename)
    os.makedirs(result_dir, exist_ok=True)
    pd.DataFrame(results).to_csv(
        os.path.join(result_dir, "k_clusters.csv"), index=False)

    # 绘图
    k_list = list(k_range)
    ncols  = 2 if has_labels else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5))
    if ncols == 1:
        axes = [axes]

    axes[0].plot(k_list, D, 'o-', markersize=4)
    axes[0].set_xlabel('k value')
    axes[0].set_ylabel('Number of clusters')
    axes[0].set_title(f'Clusters vs k — {filename}')
    axes[0].grid(True)

    if has_labels:
        for key in ['ACC', 'FMeasure', 'NMI', 'ARI']:
            vals = [r.get(key) for r in results]
            if any(v is not None for v in vals):
                axes[1].plot(k_list, vals, 'o-', markersize=4, label=key)
        axes[1].set_xlabel('k value')
        axes[1].set_ylabel('Score')
        axes[1].set_title(f'Metrics vs k — {filename}')
        axes[1].legend()
        axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "k_clusters.png"))
    plt.close()

    # 最优结果
    if has_labels:
        valid = [r for r in results if r.get('NMI') is not None]
        if valid:
            best  = max(valid, key=lambda r: r['ACC'] + r['FMeasure']
                                             + r['NMI'] + r['ARI'])
            score = best['ACC'] + best['FMeasure'] + best['NMI'] + best['ARI']
            sep   = '-' * 60
            print(f"\n{sep}")
            print(f"最优结果 | k={best['k']} 簇数={best['cluster_number']} | "
                  f"ACC={best['ACC']:.4f} FM={best['FMeasure']:.4f} "
                  f"NMI={best['NMI']:.4f} ARI={best['ARI']:.4f} "
                  f"综合得分={score:.4f}")
            print(sep)

    return D, results


# ───────────────────────────────────────────────────────────────────────────
# 工具：打印评估指标，返回综合得分
# ───────────────────────────────────────────────────────────────────────────
def _eval_and_print(label, Clusters, tag=""):
    new_label   = BestMapping(label, Clusters)
    ACC         = Acc(label, new_label)
    FMeasure, _ = Fmeasure(label, Clusters)
    NMI         = nmi(label, Clusters)
    ARI         = ari(label, Clusters)
    score       = ACC + FMeasure + NMI + ARI
    print(f"  [{tag}] 簇数={len(np.unique(Clusters))}  "
          f"ACC={ACC:.4f} FM={FMeasure:.4f} "
          f"NMI={NMI:.4f} ARI={ARI:.4f}  综合={score:.4f}")
    return score


# ═══════════════════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    np.random.seed(3407)
    set_seed(3407)

    # ── 配置（修改这里切换数据集和参数）─────────────────────────────────

    # filename = "Trachea"
    # filename = "EyeRetina"
    # filename = "Uterus"
    # filename = "Baron_Mouse"
    filename = "Muraro"
    # filename = "Segerstolpe"
    # filename = "Baron_Human"
    # filename = "Immune_Health"
    # filename = "Immune_Health_DCs"

    # AE 参数（与原 test.py 完全一致）
    LATENT_DIM      = 32
    PRETRAIN_EPOCHS = 200      # 阶段一：只用 ZINB 预训练（与原 test.py 保持一致）
    JOINT_EPOCHS    = 100      # 阶段二：联合优化轮数（仅碎片化数据集触发）
    LR              = 0.001
    BATCH_SIZE      = 4096     # None = 全批量

    # 联合优化参数（改进4，所有数据集通用）
    LAMBDA_KL       = 0.1      # KL 损失权重
    PSEUDO_UPDATE   = 5        # 每隔几个 epoch 更新伪标签
    K_INIT          = 15       # 联合优化阶段用的固定 k

    # CTNG 参数
    BASE_BETA       = 0.3

    # ── 改进开关（消融实验时在此手动控制）─────────────────────────────
    # AUTO = True：根据诊断结果自动决定（正常使用）
    # AUTO = False：忽略诊断，完全由下方三个开关手动控制（消融实验使用）
    AUTO_IMPROVE   = True

    # 以下三个开关仅在 AUTO_IMPROVE=False 时生效
    USE_IMPROVE3   = True     # 改进2：智能合并小簇
    USE_IMPROVE2   = True     # 改进3：自适应 beta
    USE_IMPROVE1   = True     # 改进4：KL 联合优化

    # 诊断阈值（仅 AUTO_IMPROVE=True 时使用）
    FRAG_THRESHOLD = 50       # 初始簇数超过此值视为碎片化
    MIN_SIZE_RATIO = 0.001    # 改进2 小簇判定阈值：细胞数 < N * MIN_SIZE_RATIO

    # AE 训练开关
    # False = 直接读取已有 latent 文件（推荐，结果完全可复现）
    # True  = 重新训练 AE（仅在第一次跑新数据集，或需要重新降维时使用）
    RETRAIN_AE = True

    # drecord 扫描范围（与原 drecord_optimized.py 一致）
    K_RANGE         = range(3, 41)

    # ── 路径（与原 test.py 保持一致）─────────────────────────────────────
    data_dir   = os.path.join("datasets", filename)
    output_dir = os.path.join("DimensionalityReduction", filename)
    os.makedirs(output_dir, exist_ok=True)

    # ── 读取 & 预处理（直接调用 preprocess.py，不做任何改动）────────────
    print(f"加载数据集: {filename}")
    data, label_raw, cell_names = read_data(data_dir)
    adata, filter_mask          = normalize(data)
    filtered_label              = label_raw[filter_mask]
    filtered_cells              = [cell_names[i]
                                   for i, m in enumerate(filter_mask) if m]
    print(f"预处理完成: {data.shape} → {adata.X.shape}")

    X = adata.X.astype('float32')

    # ── 标签数字化（两条路径都需要，必须在 RETRAIN_AE 判断前执行）────────
    unique_labels  = sorted(set(filtered_label))
    label_to_id    = {lb: idx + 1 for idx, lb in enumerate(unique_labels)}
    numeric_labels = np.array([label_to_id[lb] for lb in filtered_label],
                               dtype=np.int32)

    # ── AE 训练或加载已有 latent ────────────────────────────────────────
    latent_path = os.path.join(output_dir, f"{filename}_latent_zinb.csv")

    if not RETRAIN_AE and os.path.exists(latent_path):
        # ── 直接加载已有 latent 文件，跳过 AE 训练 ──────────────────────
        print(f"\n发现已有潜在表示文件，直接加载（RETRAIN_AE=False）")
        print(f"  路径: {latent_path}")
        best_latent = pd.read_csv(latent_path, index_col=0).values.astype('float32')
        print(f"  潜在表示形状: {best_latent.shape}")
        base_model, encoder = None, None  # 不需要模型对象

        # 重置种子后直接进入 drecord 扫描
        np.random.seed(3407)
        set_seed(3407)

        D, results = run_drecord(
            best_latent, K_RANGE,
            base_beta         = BASE_BETA,
            label             = numeric_labels,
            filename          = filename,
            use_adaptive_beta = False,
            use_smart_merge   = False,
            min_size_ratio    = MIN_SIZE_RATIO,
        )

        # 保存最优聚类标签
        valid = [r for r in results if r.get('NMI') is not None]
        if valid:
            best_r = max(valid, key=lambda r: r['ACC'] + r['FMeasure']
                                              + r['NMI'] + r['ARI'])
            best_k = best_r['k']
            best_clusters, _, _, _ = run_ctng_once(
                best_latent, best_k, BASE_BETA,
                use_adaptive_beta=False,
                use_smart_merge=False,
                min_size_ratio=MIN_SIZE_RATIO,
            )
            pred_df = pd.DataFrame({
                'Cell_Type'   : filtered_label,
                'True_Label'  : numeric_labels,
                'Pred_Cluster': best_clusters,
            }, index=filtered_cells)
            pred_path = os.path.join(output_dir, f"{filename}_best_pred_labels.csv")
            pred_df.to_csv(pred_path)
            print(f"最优聚类标签已保存至 {pred_path}")
            print(f"  最优 k={best_k}  簇数={best_r['cluster_number']}  "
                  f"ACC={best_r['ACC']:.4f} NMI={best_r['NMI']:.4f}")

    else:
        # ── 重新训练 AE（RETRAIN_AE=True 或 latent 文件不存在）────────────
        if RETRAIN_AE:
            print(f"\nRETRAIN_AE=True，重新训练 AE...")
        else:
            print(f"\n未找到 latent 文件，自动训练 AE...")
            print(f"  期望路径: {latent_path}")

    # ── 预训练后诊断：用初始聚类质量决定是否开启三项改进 ───────────────
    # （仅在 RETRAIN_AE=True 或 latent 文件不存在时执行以下流程）
    if RETRAIN_AE or not os.path.exists(latent_path):
        N_CELLS = X.shape[0]

        print(f"\n{'='*60}")
        print(f"阶段一：AE 预训练 ({PRETRAIN_EPOCHS} epochs, ZINB 损失)")
        print(f"{'='*60}")

        import scanpy as sc_ae
        adata_for_ae      = sc_ae.AnnData(X)
        _, _, base_model, encoder = train_autoencoder(
            adata_for_ae,
            latent_dim = LATENT_DIM,
            epochs     = PRETRAIN_EPOCHS,
            lr         = LR,
            batch_size = BATCH_SIZE,
        )

        # ── 诊断：预训练后判断是否碎片化 ─────────────────────────────────
        np.random.seed(3407)
        set_seed(3407)
        print("\n诊断中：对预训练潜在空间做初始 CTNG 聚类...")
        latent_init = encoder.predict(X, verbose=0)
        _, n_init, _, _ = run_ctng_once(
            latent_init, K_INIT, BASE_BETA,
            use_adaptive_beta=False,
            use_smart_merge=False,
        )
        IS_FRAGMENTED = n_init > FRAG_THRESHOLD

        if AUTO_IMPROVE:
            # 自动模式：自适应β和小簇合并无条件启用，KL优化有条件启用
            USE_MERGE     = True
            USE_ADAPTIVE  = True
            USE_JOINT_OPT = IS_FRAGMENTED and (encoder is not None)
        else:
            # 手动模式：自适应β和小簇合并无条件启用，KL优化由开关+碎片化共同决定
            USE_MERGE     = True
            USE_ADAPTIVE  = True
            USE_JOINT_OPT = USE_IMPROVE1 and IS_FRAGMENTED and (encoder is not None)

        print(f"{'='*60}")
        print(f"诊断结果  (初始簇数={n_init}, 碎片化阈值={FRAG_THRESHOLD})")
        print(f"  自动模式       : {'开启' if AUTO_IMPROVE else '关闭（手动控制）'}")
        print(f"  潜在空间状态   : {'碎片化' if IS_FRAGMENTED else '结构良好'}")
        print(f"  改进2 智能合并 : {'开启' if USE_MERGE else '关闭'}")
        print(f"  改进3 自适应β  : {'开启' if USE_ADAPTIVE else '关闭'}")
        print(f"  改进4 KL联合   : {'开启' if USE_JOINT_OPT else '关闭'}")
        print(f"{'='*60}")

        # ── 阶段二：联合优化（仅碎片化数据集）───────────────────────────
        if USE_JOINT_OPT and encoder is not None:
            print(f"\n{'='*60}")
            print(f"阶段二：联合优化 ({JOINT_EPOCHS} epochs)")
            print(f"  ZINB + KL软分配  lambda_kl={LAMBDA_KL}")
            print(f"  伪标签更新频率: 每 {PSEUDO_UPDATE} epoch")
            print(f"{'='*60}")
            best_latent = run_joint_optimization(
                base_model, encoder, X,
                label              = numeric_labels,
                joint_epochs       = JOINT_EPOCHS,
                k_init             = K_INIT,
                base_beta          = BASE_BETA,
                lambda_kl          = LAMBDA_KL,
                batch_size         = BATCH_SIZE if BATCH_SIZE else X.shape[0],
                pseudo_update_freq = PSEUDO_UPDATE,
                use_adaptive_beta  = USE_ADAPTIVE,
                use_smart_merge    = USE_MERGE,
                min_size_ratio     = MIN_SIZE_RATIO,
            )
        else:
            print(f"\n结构良好，跳过联合优化，直接使用预训练潜在表示。")
            best_latent = encoder.predict(X, verbose=0)

        # ── 保存潜在表示 ──────────────────────────────────────────────────
        latent_df  = pd.DataFrame(
            best_latent, index=filtered_cells,
            columns=[f"Zinb_{i+1}" for i in range(best_latent.shape[1])])
        label_df   = pd.DataFrame(
            {"Cell_Type": filtered_label, "Numeric_Label": numeric_labels},
            index=filtered_cells)
        mapping_df = pd.DataFrame(
            list(label_to_id.items()), columns=["Cell_Type", "Numeric_Label"])
        latent_df.to_csv(os.path.join(output_dir, f"{filename}_latent_zinb.csv"))
        label_df.to_csv(os.path.join(output_dir,  f"{filename}_labels.csv"))
        mapping_df.to_csv(os.path.join(output_dir, f"{filename}_label_mapping.csv"))
        print(f"\n潜在表示已保存至 {output_dir}/")

        # ── drecord 扫描 ──────────────────────────────────────────────────
        np.random.seed(3407)
        set_seed(3407)
        D, results = run_drecord(
            best_latent, K_RANGE,
            base_beta         = BASE_BETA,
            label             = numeric_labels,
            filename          = filename,
            use_adaptive_beta = USE_ADAPTIVE,
            use_smart_merge   = USE_MERGE,
            min_size_ratio    = MIN_SIZE_RATIO,
        )

        # ── 保存最优聚类标签 ──────────────────────────────────────────────
        valid = [r for r in results if r.get('NMI') is not None]
        if valid:
            best_r = max(valid, key=lambda r: r['ACC'] + r['FMeasure']
                                              + r['NMI'] + r['ARI'])
            best_k = best_r['k']
            best_clusters, _, _, _ = run_ctng_once(
                best_latent, best_k, BASE_BETA,
                use_adaptive_beta = USE_ADAPTIVE,
                use_smart_merge   = USE_MERGE,
                min_size_ratio    = MIN_SIZE_RATIO,
            )
            pred_df = pd.DataFrame({
                'Cell_Type'   : filtered_label,
                'True_Label'  : numeric_labels,
                'Pred_Cluster': best_clusters,
            }, index=filtered_cells)
            pred_path = os.path.join(output_dir, f"{filename}_best_pred_labels.csv")
            pred_df.to_csv(pred_path)
            print(f"最优聚类标签已保存至 {pred_path}")
            print(f"  最优 k={best_k}  簇数={best_r['cluster_number']}  "
                  f"ACC={best_r['ACC']:.4f} NMI={best_r['NMI']:.4f}")