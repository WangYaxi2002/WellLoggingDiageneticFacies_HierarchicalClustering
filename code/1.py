import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering


def load_plyl_data():
    rawData = pd.read_csv(filepath_or_buffer='../data/BAI246_CAL_GR_AC_SP.csv', header=0, encoding='utf-8',
                          dtype=np.float32, on_bad_lines='skip')
    data = rawData.values[:, 1:5]
    return rawData


if __name__ == '__main__':
    # 看见dataFrame的所有行，不然会省略
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    data = load_plyl_data()
    data_x = data.values[:, 1:5]
    num_pipeline = Pipeline([('std_scaler', MinMaxScaler())])
    train_scale = num_pipeline.fit_transform(data_x)
    ac = AgglomerativeClustering(n_clusters=5)
    clustering = ac.fit(train_scale)
    # with open('res.csv', 'a') as fp:
    #     for i in clustering.labels_:
    #         fp.write(str(i))
    #         fp.write('\n')
    # res = list(clustering.labels_)

    df = pd.DataFrame(data_x[:, 0:2])
    df['labels'] = clustering.labels_
    df1 = df[df['labels'] == 0]
    df2 = df[df['labels'] == 1]
    df3 = df[df['labels'] == 2]
    df4 = df[df['labels'] == 3]
    df5 = df[df['labels'] == 4]

    fig = plt.figure(figsize=(9, 6))
    plt.plot(df1[0], df1[1], 'bo', label='簇1')
    plt.plot(df2[0], df2[1], 'g*', )
    plt.plot(df3[0], df3[1], 'r*')
    plt.plot(df4[0], df4[1], 'c*', )
    plt.plot(df5[0], df5[1], 'm*')

    plt.xlabel('DEPTH')
    plt.ylabel('CAL')
    plt.savefig('../img/1res1.png')
    plt.show()
