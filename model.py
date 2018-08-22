# coding: utf-8
from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
import os

class Config(object):
    """RNN配置参数"""
    file_name = 'mlp'  #保存模型文件


    train_keep_prob = 0.8  # dropout保留比例
    learning_rate = 0.001  # 学习率

    batch_size = 64  # 每批训练大小
    max_steps = 2000  # 总迭代batch数

    log_every_n = 20  # 每多少轮输出一次结果
    save_every_n = 20  # 每多少轮校验模型并保存


class Model(object):

    def __init__(self, config, variable_size):
        self.config = config
        self.variable_size = variable_size

        # 待输入的数据
        self.X = tf.placeholder(dtype=tf.float32, shape=[None, self.variable_size])  # 20
        self.Y = tf.placeholder(dtype=tf.int32, shape=[None])
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        # 全局变量
        self.global_step = tf.Variable(0, trainable=False, name="global_step")

        # Ann模型
        self.Ann()

        # 初始化session
        self.saver = tf.train.Saver()
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def Ann(self):
        """Ann模型"""

        # 词嵌入层
        Y_hot = tf.one_hot(self.Y, depth=2)  # 独热编码[1,2,3] depth=5 --> [[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0]]，此时的输入节点个数为num_classes

        with tf.name_scope("ann"):
            # 定义ann网络
            w1 = tf.Variable(tf.random_normal([self.variable_size,  512], stddev=0.5))
            b1 = tf.Variable(tf.random_normal([512]))
            output = tf.matmul(self.X, w1) + b1

            # output = tf.nn.dropout(output, keep_prob=self.keep_prob)

            w2 = tf.Variable(tf.random_normal([512, 1024], stddev=.5))
            b2 = tf.Variable(tf.random_normal([1024]))
            output = tf.nn.softmax(tf.matmul(output, w2) + b2)

            w3 = tf.Variable(tf.random_normal([1024, 2], stddev=.5))
            b3 = tf.Variable(tf.random_normal([2]))
            output = tf.nn.softmax(tf.matmul(output, w3) + b3)

        with tf.name_scope("accuracy"):
            # 准确率
            prediction = tf.equal(tf.argmax(output, 1), tf.argmax(Y_hot, 1))
            self.accuracy = tf.reduce_mean(tf.cast(prediction, dtype=tf.float32))

        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            self.mean_loss = tf.reduce_mean(tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=Y_hot, logits=output)))
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.mean_loss, global_step=self.global_step)


    def load(self, checkpoint):
        self.saver.restore(self.session, checkpoint)
        print('Restored from: {}'.format(checkpoint))

    def train(self, generate_batch, x_test, y_test ):
        with self.session as sess:
            for x_batch, y_batch in generate_batch:
                # _,test_accuracy,meanloss = self.session.run([self.optim,self.accuracy,self.mean_loss], feed_dict={self.X: x_batch, self.Y: y_batch, self.keep_prob: 1.0})
                self.optim.run(feed_dict={self.X: x_batch, self.Y: y_batch, self.keep_prob: 0.7})
                if self.global_step.eval() % self.config.log_every_n == 0:
                    mean_loss = self.mean_loss.eval(feed_dict={self.X: x_batch, self.Y: y_batch, self.keep_prob: 1.0})
                    print("step %d, training accuracy %g" % (self.global_step.eval(), mean_loss))

                if self.global_step.eval() >= self.config.max_steps:
                    # self.saver.save(sess, os.path.join(model_path, 'model'))
                    test_accuracy = self.session.run(self.accuracy, feed_dict={self.X: x_test, self.Y: y_test, self.keep_prob: 1.0})
                    print(" test accuracy :", test_accuracy)
                    break


