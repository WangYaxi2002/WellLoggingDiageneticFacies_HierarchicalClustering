import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering


# 导入原始数据
def load_plyl_data():
    # 看见dataFrame的所有行，不然会省略
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    rawData = pd.read_csv(filepath_or_buffer='../data/BAI246_CAL_GR_AC_SP.csv', header=0, encoding='utf-8',
                          dtype=np.float32, on_bad_lines='skip')
    return rawData


# 获取训练数据
def get_train_scale(data_x):
    num_pipeline = Pipeline([('std_scaler', MinMaxScaler())])
    train_scale = num_pipeline.fit_transform(data_x)
    return train_scale


# 使用层次聚类算法
def agglomerativeClustering(train_scale):
    ac = AgglomerativeClustering(distance_threshold=0, n_clusters=None)  # , distance_threshold = 0
    clustering = ac.fit(train_scale)
    # 保存聚类结果到csv文件
    # with open('../data/树形图res.csv', 'a') as fp:
    #     for i in clustering.labels_:
    #         fp.write(str(i))
    #         fp.write('\n')
    res = list(clustering.labels_)
    return clustering, ac


def plot_dendrogram(model, **kwargs):
    #  创建链接矩阵，然后绘制树状图
    #  创建每个节点的样本计数
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # 叶子节点
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count
    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)
    print(linkage_matrix)
    #  绘制相应的树状图
    dendrogram(linkage_matrix, **kwargs)


# 保存图片


if __name__ == '__main__':
    data = load_plyl_data()
    data_x = data.values[:, 1:5]
    train_scale = get_train_scale(data_x)
    clustering, model = agglomerativeClustering(train_scale)

    plot_dendrogram(model, truncate_mode='level', p=5)

    plt.savefig('../img/树形图.png')
    plt.show()
