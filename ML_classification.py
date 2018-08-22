
import pandas as pd
import matplotlib.pyplot as pl
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, accuracy_score, r2_score

# TODO：从sklearn中导入监督学习模型
from sklearn.svm import LinearSVC  # 线性核svm分类
from sklearn.tree import DecisionTreeClassifier  # 决策树
from sklearn.linear_model import LogisticRegression ,SGDClassifier  #  逻辑回归,  随机梯度下降
from sklearn.naive_bayes import GaussianNB  # 高斯朴素贝叶斯
from sklearn.neighbors import KNeighborsClassifier  # k近邻
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier,AdaBoostClassifier,GradientBoostingClassifier  # 集成学习



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

# # 标签数据转换为数值
# size_mapping = {'male':1, 'female':0}
# labels = labels.map(size_mapping)

# 对于倾斜的特征使用Log转换
skewed = ['skew', 'kurt']
features[skewed] = voice_data[skewed].apply(lambda x: np.log(x + 1))

# 对于一些特征使用归一化
# scaler = MinMaxScaler()
# numerical = ['Q25', 'Q75', 'IQR', 'sfm', 'centroid']
# features[numerical] = scaler.fit_transform(voice_data[numerical])

# 切分训练、测试数据
x_train,x_test,y_train,y_test = train_test_split(features, labels,test_size=0.2, random_state=30)

# ---------------- 模型构建 ----------------
# TODO：初始化三个模型
clf_A = LinearSVC(random_state=0)
clf_B = DecisionTreeClassifier(random_state=0)
clf_C = LogisticRegression(random_state=0)

clf_A.fit(x_train,y_train)
predicts = clf_A.predict(x_test)

# ---------------- 模型评价 ----------------
print('SVM accuracy score:',accuracy_score(y_test, predicts))





