# divide.py
import numpy as np


def divide(X, TN, knn_indices, k, alpha):
    """
    划分核心点与噪声点。

    接口变化
    ────────
    原参数 Xu（N×N 排序索引矩阵）→ 新参数 knn_indices（N×(k+1) 近似 KNN 索引）。
    divide 本身逻辑不依赖完整排序矩阵，仅使用 TN 和 alpha，
    替换参数名是为了与其余模块保持一致的数据结构。

    参数
    ────
    X           : (N, d) 数据矩阵（仅用于获取 N）
    TN          : list[list[list]]，TN[i][r] = 点 i 第 r 层紧邻索引列表
    knn_indices : (N, k+1) int，近似 KNN 索引（保留接口一致性，此函数未直接使用）
    k           : 紧邻层数
    alpha       : 核心点阈值

    返回
    ────
    cl_point : 核心点索引列表
    noise    : 噪声点索引列表
    """
    N = X.shape[0]

    noise    = []
    cl_point = []

    for i in range(N):
        # 用第 k 层 TN 的大小判断是否为核心点
        if len(TN[i][k - 1]) < alpha:
            noise.append(i)
        else:
            cl_point.append(i)

    return cl_point, noise
