import pandas as pd
import matplotlib.pyplot as pl
import numpy as np
import tensorflow as tf

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

banch_size = 64
n_banch = len(x_train) // banch_size

# inputs
X = tf.placeholder(dtype=tf.float32, shape=[None, features.shape[-1]])  # 20
Y = tf.placeholder(dtype=tf.int32, shape=[None])
keep_drop = tf.placeholder(dtype=tf.float32)

#embedding
Y_hot = tf.one_hot(Y, depth=2)  # 独热编码[1,2,3] depth=5 --> [[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0]]，此时的输入节点个数为num_classes

# 3层（feed-forward）
def neural_network():
    w1 = tf.Variable(tf.random_normal([features.shape[-1], 512], stddev=0.5))
    b1 = tf.Variable(tf.random_normal([512]))
    output = tf.matmul(X, w1) + b1

    # output = tf.nn.dropout(output, keep_prob=keep_drop)

    w2 = tf.Variable(tf.random_normal([512, 1024], stddev=.5))
    b2 = tf.Variable(tf.random_normal([1024]))
    output = tf.nn.softmax(tf.matmul(output, w2) + b2)

    w3 = tf.Variable(tf.random_normal([1024, 2], stddev=.5))
    b3 = tf.Variable(tf.random_normal([2]))
    output = tf.nn.softmax(tf.matmul(output, w3) + b3)
    return output


# 训练神经网络
def train_neural_network():
    output = neural_network()

    cost = tf.reduce_mean(tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=Y_hot, logits=output)))
    lr = tf.Variable(0.001, dtype=tf.float32, trainable=False)
    opt = tf.train.AdamOptimizer(learning_rate=lr)
    var_list = [t for t in tf.trainable_variables()]
    train_step = opt.minimize(cost, var_list=var_list)


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(200):
            # sess.run(tf.assign(lr, 0.001 * (0.97 ** epoch)))

            for banch in range(n_banch):
                voice_banch = x_train[banch * banch_size:(banch + 1) * (banch_size)]
                label_banch = y_train[banch * banch_size:(banch + 1) * (banch_size)]
                _, loss = sess.run([train_step, cost], feed_dict={X: voice_banch, Y: label_banch, keep_drop:0.7})
            if epoch%10 == 0:
                print(epoch, loss)

        # 准确率
        prediction = tf.equal(tf.argmax(output, 1), tf.argmax(Y_hot, 1))
        accuracy = tf.reduce_mean(tf.cast(prediction, dtype=tf.float32))
        accuracy = sess.run(accuracy, feed_dict={X: x_test, Y: y_test,keep_drop:1})
        print("准确率", accuracy)

        # prediction = sess.run(output, feed_dict={X: test_x})


train_neural_network()
