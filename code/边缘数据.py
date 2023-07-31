import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
rawData = pd.read_csv(filepath_or_buffer='../data/BAI246_CAL_GR_AC_SP.csv', header=0, encoding='utf-8',
                      dtype=np.float32, on_bad_lines='skip')
data_x = rawData.values[:, 1:5]
num_pipeline = Pipeline([('std_scaler', MinMaxScaler())])
train_scale = num_pipeline.fit_transform(data_x)

# 假设您已经拟合了AgglomerativeClustering模型，并获得了labels_
# X为您的输入数据

# 创建AgglomerativeClustering对象，并拟合数据
model = AgglomerativeClustering(n_clusters=5)
labels = model.fit_predict(train_scale)

# 计算每个类别的样本数量
unique_labels, counts = np.unique(labels, return_counts=True)

# 找到边缘类别的索引
margin_labels = [label for label, count in zip(unique_labels, counts)]

# 筛选出边缘数据
margin_data = train_scale[np.isin(labels, margin_labels)]

print(margin_data)
