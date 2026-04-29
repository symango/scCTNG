# AE.py
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, Dense, Dropout, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from scipy import sparse


# ─────────────────────────────────────────────
# 随机种子
# ─────────────────────────────────────────────
def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'


set_seed()


# ─────────────────────────────────────────────
# ZINB 损失层（纯 TF2，移除 TF1 兼容代码）
# ─────────────────────────────────────────────
class ZINBLossLayer(Layer):
    """
    ZINB 损失层。
    将 ZINB 负对数似然作为 Keras 自定义损失，通过 add_loss() 注入训练图。
    ridge_lambda 对零膨胀概率 pi 施加 L2 正则，防止 pi 趋向极端值。
    """

    def __init__(self, ridge_lambda=0.01, **kwargs):
        super().__init__(**kwargs)
        self.ridge_lambda = ridge_lambda

    def call(self, inputs):
        y_true, pi, theta, mean = inputs
        loss = self._zinb_neg_log_likelihood(y_true, pi, theta, mean)
        # add_loss 接受标量，取均值
        self.add_loss(tf.reduce_mean(loss))
        # 返回占位符，训练时不使用
        return tf.zeros_like(y_true[:, 0:1])

    def _zinb_neg_log_likelihood(self, y_true, pi, theta, mean):
        eps = 1e-10
        mean  = tf.clip_by_value(mean,  1e-5, 1e6)
        theta = tf.clip_by_value(theta, 1e-5, 1e6)
        pi    = tf.clip_by_value(pi,    eps,   1.0 - eps)

        # ── 负二项分布（NB）部分 ──────────────────────────────────────────
        # NB 负对数似然：-log P(y | mean, theta)
        t1 = (tf.math.lgamma(theta + eps)
              + tf.math.lgamma(y_true + 1.0)
              - tf.math.lgamma(y_true + theta + eps))
        t2 = ((theta + y_true) * tf.math.log(1.0 + mean / (theta + eps))
              + y_true * (tf.math.log(theta + eps) - tf.math.log(mean + eps)))
        nb_case = t1 + t2 - tf.math.log(1.0 - pi + eps)

        # ── 零膨胀部分 ────────────────────────────────────────────────────
        # P(y=0) = pi + (1-pi) * NB(0 | mean, theta)
        zero_nb   = tf.pow(theta / (theta + mean + eps), theta)
        zero_case = -tf.math.log(pi + (1.0 - pi) * zero_nb + eps)

        # ── 合并：y==0 用零膨胀项，y>0 用 NB 项 ──────────────────────────
        result = tf.where(y_true < 1e-8, zero_case, nb_case)

        # L2 正则（压制 pi 走向极端）
        result = result + self.ridge_lambda * tf.square(pi)
        return result

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'ridge_lambda': self.ridge_lambda})
        return cfg


# ─────────────────────────────────────────────
# 编码器 / 解码器构建
# ─────────────────────────────────────────────
def _dense_drop(x, units, dropout_rate=0.2):
    """
    Dense → relu → Dropout 的原始块结构。
    恢复原始写法，避免 BatchNormalization 改变潜在空间分布，
    影响后续 CTNG 聚类的距离计算。
    """
    x = Dense(units, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    return x


def build_autoencoder(input_dim, latent_dim=32):
    """
    构建 ZINB 自编码器。
    网络结构恢复原始的 Dense+relu+Dropout，
    仅保留 Lambda clip 修复和潜在层线性激活两处改动供对比测试。
    若聚类效果仍不理想，可将 latent 的 activation 改回 'relu'。
    """
    X_input = Input(shape=(input_dim,), name='feature_input')

    # ── 编码器 ────────────────────────────────────────────────────────────
    h = _dense_drop(X_input, 256, dropout_rate=0.2)
    h = _dense_drop(h,       128, dropout_rate=0.2)
    # 潜在层：可在 'relu' 和 'linear' 之间切换对比效果
    # 若聚类变差，优先改回 'relu'
    latent = Dense(latent_dim, activation='relu', name='latent')(h)

    # ── 解码器 ────────────────────────────────────────────────────────────
    d = _dense_drop(latent, 128, dropout_rate=0.2)
    d = _dense_drop(d,      256, dropout_rate=0.2)

    # ── ZINB 三组参数输出 ──────────────────────────────────────────────────
    pi    = Dense(input_dim, activation='sigmoid',  name='pi')(d)
    theta = Dense(input_dim, activation='softplus', name='theta_raw')(d)
    theta = Lambda(lambda x: tf.clip_by_value(x, 1e-4, 1e4), name='theta')(theta)
    mean  = Dense(input_dim, activation='softplus', name='mean_raw')(d)
    mean  = Lambda(lambda x: tf.clip_by_value(x, 1e-5, 1e6),  name='mean')(mean)

    model = Model(inputs=X_input, outputs=[pi, theta, mean, latent],
                  name='zinb_autoencoder')
    return model


# ─────────────────────────────────────────────
# 训练入口
# ─────────────────────────────────────────────
def train_autoencoder(adata, latent_dim=32, epochs=200, lr=0.001, batch_size=None):
    """
    训练 ZINB 自编码器并返回潜在表示。

    训练方式完全恢复原始全批量（batch_size=n_cells）写法。
    全批量对 ZINB 损失更稳定：每次梯度更新看到全部细胞，
    pi/theta/mean 的估计不受 mini-batch 采样偏差影响。
    对于大数据集内存不足的情况，可以手动传入较大的 batch_size 参数降级处理。
    """
    print("开始训练自编码器...")

    X = adata.X.astype(np.float32)
    if sparse.issparse(X):
        X = X.toarray()

    input_dim = X.shape[1]
    n_cells   = X.shape[0]
    print(f"数据维度: {X.shape}")

    # ── 构建模型 ──────────────────────────────────────────────────────────
    base_model = build_autoencoder(input_dim, latent_dim)

    X_input      = Input(shape=(input_dim,), name='feature_input')
    y_true_input = Input(shape=(input_dim,), name='y_true')

    pi, theta, mean, encoded = base_model(X_input)
    loss_out = ZINBLossLayer(ridge_lambda=0.01,
                             name='zinb_loss')([y_true_input, pi, theta, mean])

    training_model = Model(
        inputs=[X_input, y_true_input],
        outputs=loss_out,
        name='training_model'
    )
    training_model.compile(
        optimizer=Adam(learning_rate=lr),
        loss=lambda y_true, y_pred: 0
    )

    print("开始训练...")

    # EarlyStopping：监控训练损失（全批量无验证集，用train_loss）
    # 连续 20 个 epoch 损失下降不足 1e-4 则停止，restore_best_weights 保存最优权重
    early_stop = EarlyStopping(
        monitor='loss',
        patience=20,
        min_delta=1e-4,
        restore_best_weights=True,
        verbose=1
    )

    # batch_size 默认 None 表示全批量（n_cells）。
    # 大规模数据内存不足时，在 test.py 调用处传入如 2048、4096 等较小值。
    # 除此之外不要改动其他任何参数，避免影响聚类质量。
    actual_batch = n_cells if batch_size is None else batch_size
    mode_str = "全批量" if batch_size is None else "mini-batch"
    print(f"batch_size={actual_batch} ({mode_str})")

    history = training_model.fit(
        [X, X],
        np.zeros((n_cells, 1)),
        epochs=epochs,
        batch_size=actual_batch,
        verbose=1,
        shuffle=False,
        callbacks=[early_stop]
    )

    print("训练完成，提取潜在表示...")

    # ── 提取潜在表示 ──────────────────────────────────────────────────────
    encoder_model = Model(inputs=X_input, outputs=encoded, name='encoder')
    latent_repr   = encoder_model.predict(X)

    print(f"降维完成，潜在表示形状: {latent_repr.shape}")

    # 同时返回 base_model 和 encoder_model，供 run.py 联合优化阶段复用。
    # 原有调用 latent_repr, history = train_autoencoder(...) 不再兼容，
    # 请改为 latent_repr, history, base_model, encoder_model = train_autoencoder(...)
    return latent_repr, history, base_model, encoder_model