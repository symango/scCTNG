# improvements.py
# 包含 AE_CTNG 的三项改进：
#   改进2：小簇智能合并后处理
#   改进3：beta 局部密度自适应
#   改进4：KL 软分配联合优化（类 scDeepCluster）

import numpy as np
import tensorflow as tf
from scipy.spatial.distance import cdist


# ═══════════════════════════════════════════════════════════════════════════
# 改进3：beta 局部密度自适应
# ═══════════════════════════════════════════════════════════════════════════
def compute_adaptive_beta(knn_distances, base_beta=0.3,
                          min_beta=0.15, max_beta=0.7):
    """
    为每个点计算自适应 beta 值，取全局中位数作为本次聚类的 beta。

    原理：局部密度 = 1 / mean(k近邻距离)
         密度越高 → beta 越大（合并更保守，避免不同簇被粘连）
         密度越低 → beta 越小（合并更宽松，帮助稀疏区域连通）

    参数
    ────
    knn_distances : (N, k+1) float，build_knn_index 返回的距离矩阵
    base_beta     : 基准 beta，归一化中心点
    min_beta      : beta 下界
    max_beta      : beta 上界

    返回
    ────
    beta_median : float，自适应 beta 的全局中位数，直接传入 clustering()
    beta_arr    : (N,) float，每个点的自适应 beta（供分析/可视化用）
    """
    mean_dist = knn_distances[:, 1:].mean(axis=1)   # 去掉第0列自身
    mean_dist = np.clip(mean_dist, 1e-10, None)
    density   = 1.0 / mean_dist

    d_min, d_max = density.min(), density.max()
    if d_max - d_min < 1e-10:
        beta_arr = np.full(len(density), base_beta, dtype=np.float32)
    else:
        density_norm = (density - d_min) / (d_max - d_min)
        beta_arr = (min_beta + density_norm * (max_beta - min_beta)
                    ).astype(np.float32)

    beta_median = float(np.median(beta_arr))
    return beta_median, beta_arr


# ═══════════════════════════════════════════════════════════════════════════
# 改进2：小簇智能合并后处理
# ═══════════════════════════════════════════════════════════════════════════
def smart_merge_small_clusters(Clusters, X_latent, min_size_ratio=0.01):
    """
    将过小的连通分支合并到距离最近的大簇。

    原 clustering.py 中小于 3 个点的分支直接归为噪声再分配，
    但对于过渡态数据集，噪声分配逻辑仍会留下大量小碎片簇。
    本函数在 clustering() 之后做一次后处理：
        1. 识别小于 N * min_size_ratio 的簇（至少 4 个点）
        2. 计算小簇质心到所有大簇质心的欧氏距离
        3. 将小簇整体合并到最近的大簇

    参数
    ────
    Clusters       : (N,) int，clustering() 返回的聚类标签
    X_latent       : (N, d) float，潜在空间表示
    min_size_ratio : 小于 N * min_size_ratio 的簇视为小簇，默认 1%

    返回
    ────
    new_clusters : (N,) int，合并后重新编号的聚类标签
    """
    N        = len(Clusters)
    min_size = max(4, int(N * min_size_ratio))
    labels   = np.array(Clusters)

    unique_cls     = np.unique(labels)
    big_clusters   = [c for c in unique_cls if np.sum(labels == c) >= min_size]
    small_clusters = [c for c in unique_cls if np.sum(labels == c) < min_size]

    if not small_clusters or not big_clusters:
        return Clusters

    big_centroids = np.array(
        [X_latent[labels == c].mean(axis=0) for c in big_clusters])

    new_labels = labels.copy()
    for sc_id in small_clusters:
        mask     = labels == sc_id
        centroid = X_latent[mask].mean(axis=0, keepdims=True)
        dists    = cdist(centroid, big_centroids, metric='euclidean')[0]
        nearest  = big_clusters[int(np.argmin(dists))]
        new_labels[mask] = nearest

    # 重新编号为连续整数（从 1 开始，与原代码保持一致）
    mapping = {old: new + 1
               for new, old in enumerate(np.unique(new_labels))}
    return np.array([mapping[l] for l in new_labels], dtype=np.int32)


# ═══════════════════════════════════════════════════════════════════════════
# 改进4：KL 软分配联合优化
# ═══════════════════════════════════════════════════════════════════════════
def soft_assignment(latent, cluster_centers):
    """
    Student-t 分布软分配 q_{ij}（DEC/scDeepCluster 同款）。

    参数
    ────
    latent          : (N, d) numpy float32
    cluster_centers : (K, d) numpy float32

    返回
    ────
    q : (N, K) numpy float32
    """
    diff  = latent[:, np.newaxis, :] - cluster_centers[np.newaxis, :, :]
    dist2 = np.sum(diff ** 2, axis=2)
    q     = 1.0 / (1.0 + dist2)
    q     = q / q.sum(axis=1, keepdims=True)
    return q.astype(np.float32)


def target_distribution(q):
    """
    辅助目标分布 p_{ij} = (q²/freq) / Σ(q²/freq)
    强化高置信度分配，压制低置信度分配。
    """
    freq = q.sum(axis=0, keepdims=True)
    p    = q ** 2 / (freq + 1e-10)
    p    = p / p.sum(axis=1, keepdims=True)
    return p.astype(np.float32)


def kl_loss_value(p, q):
    """
    KL(P || Q) 标量，用于联合优化的辅助损失。
    p, q : (batch, K) tf.Tensor 或 numpy
    """
    p = tf.cast(p, tf.float32)
    q = tf.cast(q, tf.float32)
    p = tf.clip_by_value(p, 1e-10, 1.0)
    q = tf.clip_by_value(q, 1e-10, 1.0)
    return tf.reduce_mean(tf.reduce_sum(p * tf.math.log(p / q), axis=1))


def joint_optimize(base_model, encoder, X, pseudo_labels, cluster_centers,
                   optimizer, lambda_kl=0.1, batch_size=4096):
    """
    执行一轮联合优化：ZINB 损失 + KL 软分配损失。

    说明
    ────
    • ZINB 损失通过 base_model 的 add_loss 自动注入，不需要单独计算。
    • KL 损失依赖 pseudo_labels 推导出的软分配目标分布 p。
    • 每次调用只跑一个 epoch，外部 for loop 控制总轮数。

    参数
    ────
    base_model      : build_autoencoder 返回的模型（含 ZINB 输出头）
    encoder         : 只输出 latent 的子模型
    X               : (N, input_dim) float32 预处理后数据
    pseudo_labels   : (N,) int，当前伪标签（0-indexed）
    cluster_centers : (K, d) float32，当前簇中心
    optimizer       : tf.keras.optimizers 实例
    lambda_kl       : KL 损失权重
    batch_size      : mini-batch 大小

    返回
    ────
    epoch_loss : float，本轮平均总损失
    """
    n_cells   = X.shape[0]
    n_batches = max(1, n_cells // batch_size)
    epoch_loss = 0.0

    # 提前计算全局软分配 q 和目标分布 p（避免 batch 内重复计算）
    latent_np = encoder.predict(X, verbose=0)
    q_np      = soft_assignment(latent_np, cluster_centers)
    p_np      = target_distribution(q_np)

    for b in range(n_batches):
        start = b * batch_size
        end   = min(start + batch_size, n_cells)
        X_b   = tf.constant(X[start:end], dtype=tf.float32)
        p_b   = tf.constant(p_np[start:end])

        with tf.GradientTape() as tape:
            pi_b, theta_b, mean_b, latent_b = base_model(X_b, training=True)

            # ── ZINB 损失（直接在 tape 内计算，避免 add_loss 梯度追踪问题）──
            eps   = 1e-10
            mean_ = tf.clip_by_value(mean_b,  1e-5, 1e6)
            tht_  = tf.clip_by_value(theta_b, 1e-5, 1e6)
            pi_   = tf.clip_by_value(pi_b,    eps,  1.0 - eps)

            t1 = (tf.math.lgamma(tht_ + eps)
                  + tf.math.lgamma(X_b + 1.0)
                  - tf.math.lgamma(X_b + tht_ + eps))
            t2 = ((tht_ + X_b) * tf.math.log(1.0 + mean_ / (tht_ + eps))
                  + X_b * (tf.math.log(tht_ + eps) - tf.math.log(mean_ + eps)))
            nb_case   = t1 + t2 - tf.math.log(1.0 - pi_ + eps)
            zero_nb   = tf.pow(tht_ / (tht_ + mean_ + eps), tht_)
            zero_case = -tf.math.log(pi_ + (1.0 - pi_) * zero_nb + eps)
            zinb_elem = tf.where(X_b < 1e-8, zero_case, nb_case)
            zinb_l    = tf.reduce_mean(zinb_elem + 0.01 * tf.square(pi_))

            # ── KL 损失（全部保持在 tf 计算图内，不调用 .numpy()）──────────
            cc_tf  = tf.constant(cluster_centers, dtype=tf.float32)
            diff   = tf.expand_dims(latent_b, 1) - tf.expand_dims(cc_tf, 0)
            dist2  = tf.reduce_sum(diff ** 2, axis=2)
            q_new  = 1.0 / (1.0 + dist2)
            q_new  = q_new / tf.reduce_sum(q_new, axis=1, keepdims=True)
            kl_l   = kl_loss_value(p_b, q_new)

            total = zinb_l + lambda_kl * kl_l

        grads = tape.gradient(total, base_model.trainable_variables)
        # 过滤掉 None 梯度（部分输出头在当前 batch 可能无贡献）
        grads_and_vars = [(g, v) for g, v in zip(grads, base_model.trainable_variables)
                          if g is not None]
        optimizer.apply_gradients(grads_and_vars)
        epoch_loss += float(total)

    return epoch_loss / n_batches