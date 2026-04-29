import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score


def Acc(true_labels, pred_labels):
    """
    计算准确率
    参数:
        true_labels: 真实标签
        pred_labels: 预测标签
    返回:
        acc: 准确率
    """
    return np.sum(true_labels == pred_labels) / len(true_labels)


def Fmeasure(true_labels, pred_labels):
    """
    计算F-measure
    参数:
        true_labels: 真实标签
        pred_labels: 预测标签
    返回:
        FMeasure: F-measure值
        Accuracy: 准确率
    """
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)

    N = len(pred_labels)
    p_classes = np.unique(true_labels)
    c_classes = np.unique(pred_labels)

    P_size = len(p_classes)
    C_size = len(c_classes)

    # 构建指示矩阵
    Pid = np.zeros((P_size, N))
    Cid = np.zeros((C_size, N))

    for i in range(P_size):
        Pid[i, :] = (true_labels == p_classes[i])

    for i in range(C_size):
        Cid[i, :] = (pred_labels == c_classes[i])

    # 计算交集
    CP = np.dot(Cid, Pid.T)
    Pj = np.sum(CP, axis=0)
    Ci = np.sum(CP, axis=1)

    # 计算精确率和召回率
    precision = CP / (Ci[:, np.newaxis] + 1e-10)
    recall = CP / (Pj + 1e-10)

    # 计算F值
    F = 2 * precision * recall / (precision + recall + 1e-10)

    # 计算总的F值
    F_max = np.max(F, axis=0)
    FMeasure = np.sum((Pj / np.sum(Pj)) * F_max)

    # 计算准确率
    Accuracy = np.sum(np.max(CP, axis=0)) / N

    return FMeasure, Accuracy


def nmi(true_labels, pred_labels):
    """
    计算标准化互信息
    参数:
        true_labels: 真实标签
        pred_labels: 预测标签
    返回:
        nmi: 标准化互信息值
    """
    return normalized_mutual_info_score(true_labels, pred_labels)


def ari(true_labels, pred_labels):
    """
    计算调整兰德指数
    参数:
        true_labels: 真实标签
        pred_labels: 预测标签
    返回:
        ari: 调整兰德指数值
    """
    return adjusted_rand_score(true_labels, pred_labels)