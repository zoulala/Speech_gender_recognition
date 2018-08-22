import os
import pandas as pd
import matplotlib.pyplot as pl
import numpy as np
import tensorflow as tf
from model import  Model,Config

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, accuracy_score, r2_score



# ---------------- 读入数据 ----------------
voice_data = pd.read_csv('voice.csv')
print(voice_data.shape)
print(voice_data.head())

# # ---------------- 数据展示 ----------------
# pl.figure(figsize=(11, 5))
# pl.plot(voice_data['skew'], 'r-', label = 'skew')
# pl.legend()
# pl.xlabel('samples')          # 设置x轴的label，pyplot模块提供了很直接的方法，内部也是调用的上面当然讲述的面向对象的方式来设置；
# pl.ylabel('value')      # 设置y轴的label;
# pl.show()

# ---------------- 数据预处理 ----------------
# 分离出特征、标签数据
labels = voice_data['label']
features = voice_data.drop('label', axis = 1)

# 标签数据转换为数值
size_mapping = {'male':1, 'female':0}
labels = labels.map(size_mapping)


# 对于倾斜的特征使用Log转换
skewed = ['skew', 'kurt']
features[skewed] = voice_data[skewed].apply(lambda x: np.log(x + 1))

# 对于一些特征使用归一化
# scaler = MinMaxScaler()
# numerical = ['Q25', 'Q75', 'IQR', 'sfm', 'centroid']
# features[numerical] = scaler.fit_transform(voice_data[numerical])


# 切分训练、测试数据
x_train,x_test,y_train,y_test = train_test_split(features, labels,test_size=0.2, random_state=30)


# ---------------- 多层感知机(MLP) ----------------




model = Model(Config, features.shape[-1])


def batch_generator(x ,y , batchsize):
    '''产生训练batch样本'''
    assert len(x)==len(y), 'error:len_x != len_y'
    n_samples = len(y)
    n_batches = int(n_samples/batchsize)
    n = n_batches * batchsize
    while True:
        for i in range(0, n, batchsize):
            voice_banch = x[i:i +batchsize]
            label_banch = y[i:i +batchsize]
            yield voice_banch,label_banch


model.train( batch_generator(x_train,y_train,Config.batch_size), x_test,y_test)



