# preprocess.py
import pandas as pd
import numpy as np
import scanpy as sc
from scipy import sparse
import warnings

warnings.filterwarnings('ignore')


def read_data(filepath):
    """读取数据和标签"""
    try:
        # 读取data.csv
        # 大规模数据（>1万细胞）直接 read_csv 会 MemoryError，
        # 改用 chunksize 分块读取，每次只加载一部分行，
        # 读完后拼接为 float32，避免中间产生 float64 的全量副本。
        data_path = f"{filepath}/data.csv"
        chunks = pd.read_csv(data_path, header=0, index_col=0,
                             chunksize=2000)
        chunk_list = []
        cell_names = []
        for chunk in chunks:
            cell_names.extend(chunk.index.tolist())
            chunk_list.append(chunk.values.astype(np.float32))
        data = np.vstack(chunk_list)

        # 读取label.csv - 修复标签读取逻辑
        label_path = f"{filepath}/label.csv"
        df_label = pd.read_csv(label_path)

        # 检查列名是否为cluster
        if 'cluster' in df_label.columns:
            label = df_label['cluster'].values
        else:
            # 如果没有cluster列，则取第一列
            label = df_label.iloc[:, 0].values

        # 确保数据和标签长度匹配
        if len(data) != len(label):
            raise ValueError(f"数据和标签长度不匹配: 数据长度={len(data)}, 标签长度={len(label)}")

        return data, label, cell_names

    except Exception as e:
        print(f"读取数据时出错: {str(e)}")
        raise


def normalize(data, min_cells=3, n_gene_thresholds=(0.01, 0.99),
              mito_threshold=0.95, highly_genes=2000):
    """标准化处理，基于Seurat方法"""
    # 创建AnnData对象
    adata = sc.AnnData(sparse.csr_matrix(data)) if sparse.issparse(data) else sc.AnnData(data)

    # 计算基础指标
    adata.obs["n_counts"] = np.array(adata.X.sum(axis=1)).flatten()

    # 检测线粒体基因
    mt_prefixes = ['mt-', 'MT-', 'Mt-', 'mMT-', 'MT_', 'mt_', 'mito-', 'MITO-']
    mito_genes = adata.var_names.str.startswith(tuple(mt_prefixes))

    if mito_genes.sum() > 0:
        adata.obs["percent_mt"] = np.array(
            adata[:, mito_genes].X.sum(axis=1)
        ).flatten() / adata.obs["n_counts"] * 100
        print(f"检测到{mito_genes.sum()}个线粒体基因")
    else:
        adata.obs["percent_mt"] = 0.0
        print("警告：未检测到线粒体基因，将跳过线粒体过滤条件")

    # 基因过滤
    print(f"原始基因数: {adata.n_vars}")
    sc.pp.filter_genes(adata, min_cells=min_cells)
    print(f"基因过滤后: {adata.n_vars}个基因 (min_cells={min_cells})")

    # 计算过滤后的基因数指标
    adata.obs["n_genes"] = np.array((adata.X > 0).sum(axis=1)).flatten()

    # 动态计算过滤阈值
    n_genes_low = np.percentile(adata.obs["n_genes"], n_gene_thresholds[0] * 100)
    n_genes_high = np.percentile(adata.obs["n_genes"], n_gene_thresholds[1] * 100)

    # 根据是否检测到线粒体基因调整过滤条件
    if mito_genes.sum() > 0:
        mito_threshold_val = np.percentile(adata.obs["percent_mt"], mito_threshold * 100)
        print(f"基因数阈值: {n_genes_low:.1f}-{n_genes_high:.1f}")
        print(f"线粒体阈值: <{mito_threshold_val:.2f}%")

        cell_filter = (
                (adata.obs["n_genes"] > n_genes_low) &
                (adata.obs["n_genes"] < n_genes_high) &
                (adata.obs["percent_mt"] < mito_threshold_val))
    else:
        print(f"基因数阈值: {n_genes_low:.1f}-{n_genes_high:.1f}")
        print("跳过线粒体过滤条件")

        cell_filter = (
                (adata.obs["n_genes"] > n_genes_low) &
                (adata.obs["n_genes"] < n_genes_high))

    # 执行过滤
    adata = adata[cell_filter, :].copy()
    filter_mask = cell_filter.values
    print(f"细胞过滤后: {adata.n_obs}个细胞")

    # 标准化流程
    if not sparse.issparse(adata.X) and adata.n_obs > 1000:
        adata.X = sparse.csr_matrix(adata.X)

    # 标准化
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # 高变基因筛选
    try:
        if highly_genes and highly_genes > 0 and highly_genes < adata.n_vars:
            print(f"筛选{highly_genes}个高变基因")
            sc.pp.highly_variable_genes(
                adata,
                n_top_genes=highly_genes,
                subset=True
            )
        else:
            print(f"使用cutoff筛选高变基因 (当前基因数: {adata.n_vars})")
            sc.pp.highly_variable_genes(
                adata,
                min_mean=0.01,
                max_mean=3,
                min_disp=0.5,
                subset=True
            )
    except Exception as e:
        print(f"高变基因筛选失败：{str(e)}，使用全部基因")
        adata.var["highly_variable"] = True

    # 缩放数据
    sc.pp.scale(adata, max_value=10)

    # 转换为稠密矩阵
    if sparse.issparse(adata.X):
        adata.X = adata.X.toarray()
    elif not isinstance(adata.X, np.ndarray):
        adata.X = np.array(adata.X)

    return adata, filter_mask