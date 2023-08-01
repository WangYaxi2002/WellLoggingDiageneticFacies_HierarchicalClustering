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
    rawData = pd.read_csv(filepath_or_buffer='../../data/BAI246_CAL_GR_AC_SP.csv', header=0, encoding='utf-8',
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
    clustering = ac.fit(train_scale)
    return clustering


# 保存图片
def saveImage(data, res):
    labels = ['1', '2', '3', '4', '5']
    colors = [(1, 0, 0,), (0, 1, 0), (0, 0, 1), (1, 1, 0), (0, 1, 1)]
    df = pd.DataFrame(data)
    df['labels'] = res
    df1 = df[df['labels'] == 0]
    df2 = df[df['labels'] == 1]
    df3 = df[df['labels'] == 2]
    df4 = df[df['labels'] == 3]
    df5 = df[df['labels'] == 4]
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    x_position_fmt = ['CAL', 'AC', 'GR', 'SP']
    x_position = [1, 3.5, 6, 8.5]
    for i in range(4):
        bplot = plt.boxplot(
            [df1.loc[:, x_position_fmt[i]], df2.loc[:, x_position_fmt[i]], df3.loc[:, x_position_fmt[i]],
             df4.loc[:, x_position_fmt[i]], df5.loc[:, x_position_fmt[i]]],
            patch_artist=True,
            labels=labels,
            positions=(1 + 2.5 * i, 1.4 + 2.5 * i, 1.8 + 2.5 * i, 2.2 + 2.5 * i, 2.6 + 2.5 * i),
            widths=0.3)
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)

    plt.xticks([i + 0.8 / 2 for i in x_position], x_position_fmt)

    legend_handles = []
    for color, label in zip(colors, labels):
        legend_handles.append(mpatches.Patch(color=color, label=label))
    plt.legend(handles=legend_handles, title='聚类类别')

    plt.xlabel('参数')
    plt.ylabel('原始数据')
    plt.savefig('../../img/带有异常值箱型图/按参数的原始数据箱型图.png')
    plt.show()


if __name__ == '__main__':
    data = load_plyl_data()
    data_x = data.values[:, 1:5]
    train_scale = get_train_scale(data_x)
    clustering = agglomerativeClustering(train_scale)
    saveImage(data, list(clustering.labels_))
