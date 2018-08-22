import pandas as pd
import matplotlib.pyplot as pl
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, accuracy_score, r2_score

from xgboost.sklearn import XGBClassifier



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

# ---------------- 模型构建 ----------------
clf_A = XGBClassifier(
 learning_rate =0.3, #默认0.3
 n_estimators=2000, #树的个数
 max_depth=3,# 构建树的深度，越大越容易过拟合
 min_child_weight=1,# 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言,假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting
 gamma=0.5,# 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子
 subsample=0.6,# 随机采样训练样本
 colsample_bytree=0.6,# 生成树时进行的列采样
 objective= 'binary:logistic', #逻辑回归损失函数
 nthread=4,  #cpu线程数
 scale_pos_weight=1,
 reg_alpha=1e-05,
 reg_lambda=1,
 seed=27)


clf_A.fit(x_train,y_train)
predicts = clf_A.predict(x_test)

# ---------------- 模型评价 ----------------
print('XGB accuracy score:',accuracy_score(y_test, predicts))