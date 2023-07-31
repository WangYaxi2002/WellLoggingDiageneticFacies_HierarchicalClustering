import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering
import matplotlib.patches as mpatches


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
    ac = AgglomerativeClustering(n_clusters=5)
    clustering = ac.fit(train_scale[:, :4])
    # 保存聚类结果到csv文件
    # with open('res.csv', 'a') as fp:
    #     for i in clustering.labels_:
    #         fp.write(str(i))
    #         fp.write('\n')
    # res = list(clustering.labels_)
    return clustering


# 保存图片
def saveImage(data, res):
    labels = ['CAL', 'AC', 'GR', 'SP']
    colors = [(1, 0, 0,), (0, 1, 0), (0, 0, 1), (1, 1, 0)]
    df = pd.DataFrame(data)
    df['labels'] = res
    df1 = df[df['labels'] == 0]
    df2 = df[df['labels'] == 1]
    df3 = df[df['labels'] == 2]
    df4 = df[df['labels'] == 3]
    df5 = df[df['labels'] == 4]
    bplot1 = plt.boxplot(df1.values[:, :4], showfliers=False, patch_artist=True, labels=labels,
                         positions=(1, 1.3, 1.6, 1.9), widths=0.3)
    for patch, color in zip(bplot1['boxes'], colors):
        patch.set_facecolor(color)

    bplot2 = plt.boxplot(df2.values[:, :4], showfliers=False, patch_artist=True, labels=labels,
                         positions=(2.5, 2.8, 3.1, 3.4),
                         widths=0.3)

    for patch, color in zip(bplot2['boxes'], colors):
        patch.set_facecolor(color)

    bplot3 = plt.boxplot(df3.values[:, :4], showfliers=False, patch_artist=True, labels=labels,
                         positions=(4, 4.3, 4.6, 4.9), widths=0.3)

    for patch, color in zip(bplot3['boxes'], colors):
        patch.set_facecolor(color)

    bplot4 = plt.boxplot(df4.values[:, :4], showfliers=False, patch_artist=True, labels=labels,
                         positions=(5.5, 5.8, 6.1, 6.4),
                         widths=0.3)

    for patch, color in zip(bplot4['boxes'], colors):
        patch.set_facecolor(color)

    bplot5 = plt.boxplot(df5.values[:, :4], showfliers=False, patch_artist=True, labels=labels,
                         positions=(7, 7.3, 7.6, 7.9), widths=0.3)

    for patch, color in zip(bplot5['boxes'], colors):
        patch.set_facecolor(color)

    x_position = [1, 2.5, 4, 5.5, 7]
    x_position_fmt = ['1', '2', '3', '4', '5']

    legend_handles = []
    for color, label in zip(colors, labels):
        legend_handles.append(mpatches.Patch(color=color, label=label))
    plt.legend(handles=legend_handles)
    plt.xticks([i + 0.8 / 2 for i in x_position], x_position_fmt)
    plt.savefig('../img/归一化数据箱型图.png')
    plt.show()


if __name__ == '__main__':
    data = load_plyl_data()
    data_x = data.values[:, 1:5]
    train_scale = get_train_scale(data_x)
    with open('train', 'a') as fp:
        for i in train_scale:
            for j in i:
                fp.write(str(j) + "  ")
            fp.write("\n")

    clustering = agglomerativeClustering(train_scale)
    saveImage(train_scale, list(clustering.labels_))
