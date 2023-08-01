# 使用曼哈顿距离算法聚类
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering


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
    ac = AgglomerativeClustering(n_clusters=5, metric='manhattan', linkage='complete')
    clustering = ac.fit(train_scale)

    # 保存聚类结果到csv文件
    # with open('res.csv', 'a') as fp:
    #     for i in clustering.labels_:
    #         fp.write(str(i))
    #         fp.write('\n')
    # res = list(clustering.labels_)
    return clustering


# 保存图片
def saveImage(labels: list):
    # 对6种组合（12种结果）保存图片
    for i in range(4):
        for j in range(4):
            if i == j:
                continue
            df = pd.DataFrame(data_x[:, [i, j]])
            df['labels'] = labels
            df1 = df[df['labels'] == 0]
            df2 = df[df['labels'] == 1]
            df3 = df[df['labels'] == 2]
            df4 = df[df['labels'] == 3]
            df5 = df[df['labels'] == 4]
            plt.figure(figsize=(10, 10))
            plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
            plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
            headers = ['1', '2', '3', '4', '5']
            p1, = plt.plot(df1[0], df1[1], 'bo', label='1')
            p2, = plt.plot(df2[0], df2[1], 'g*', label='2')
            p3, = plt.plot(df3[0], df3[1], 'r*', label='3')
            p4, = plt.plot(df4[0], df4[1], 'c*', label='4')
            p5, = plt.plot(df5[0], df5[1], 'm*', label='5')
            plt.legend((p1, p2, p3, p4, p5), headers, title='聚类类别')
            plt.xlabel(data.columns.values[i + 1])
            plt.ylabel(data.columns.values[j + 1])
            plt.savefig(
                '../../img/散点图/曼哈顿距离' + data.columns.values[i + 1] + '-' + data.columns.values[j + 1] + '.png')
            plt.show()


if __name__ == '__main__':
    data = load_plyl_data()
    data_x = data.values[:, 1:5]
    train_scale = get_train_scale(data_x)
    clustering = agglomerativeClustering(train_scale)
    saveImage(list(clustering.labels_))
