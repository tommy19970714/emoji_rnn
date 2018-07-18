#!/usr/bin/env python
# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import collections
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split

# ベクトルの読み込み
data = pickle.loads(open('./text_vec.pkl', 'rb').read())
label = pickle.loads(open('./label_vec.pkl', 'rb').read())
label_count = len(collections.Counter([str(v) for v in label]))

# データ作成
X = []
y = []

# データ数を合わせる
c = collections.Counter(label)
sample_nums = c.most_common()
print("sample_nums:", sample_nums)

min_num = np.min([s[1] for s in sample_nums])
print("min_num:", min_num)

for sample in sample_nums:
    diff_num = int(sample[1] - min_num)
    print("クラス%d 削除サンプル数: %d (%0.2f％)" % (sample[0], diff_num, (diff_num/sample[1])*100))
    indexes = [i for i, l in enumerate(label) if l == sample[0]]
    del_indexes = random.sample(indexes, min_num)
    X.extend([data[i] for i in indexes])
    y.extend([label[i] for i in indexes])
X = np.array(X)
y = np.array(y)

# パラメーター
N_CLASSES = label_count # クラス数
N_INPUTS = 200  # 1ステップに入力されるデータ数
N_STEPS = 250  # 学習ステップ数
LEN_SEQ = 30  # 系列長
N_NODES = 100  # ノード数
BATCH_SIZE = 36  # バッチサイズ


print("length")
print(len(X))
print(len(y))

x_train, x_test, t_train, t_test = train_test_split(X, y, test_size=0.20, random_state=random.randint(0, 100))

N_DATA = len(x_train)  # 各クラスの学習用データ数
N_TEST = len(x_test)  # テスト用データ数

print("N_CLASSES = %s" % N_CLASSES)
print("N_DATA = %s" % N_DATA)


# モデルの構築
x = tf.placeholder(tf.float32, [None, LEN_SEQ, N_INPUTS])  # 入力データ
t = tf.placeholder(tf.int32, [None])  # 教師データ
t_on_hot = tf.one_hot(t, depth=N_CLASSES, dtype=tf.float32)  # 1-of-Kベクトル
cell = rnn.BasicRNNCell(num_units=N_NODES, activation=tf.nn.tanh)  # 中間層のセル
# RNNに入力およびセル設定する
outputs, states = tf.nn.dynamic_rnn(cell=cell, inputs=x, dtype=tf.float32, time_major=False)
# [ミニバッチサイズ,系列長,出力数]→[系列長,ミニバッチサイズ,出力数]
outputs = tf.transpose(outputs, perm=[1, 0, 2])

w = tf.Variable(tf.random_normal([N_NODES, N_CLASSES], stddev=0.01))
b = tf.Variable(tf.zeros([N_CLASSES]))
logits = tf.matmul(outputs[-1], w) + b  # 出力層
pred = tf.nn.softmax(logits)  # ソフトマックス

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=t_on_hot, logits=logits)
loss = tf.reduce_mean(cross_entropy)  # 誤差関数
train_step = tf.train.AdamOptimizer().minimize(loss)  # 学習アルゴリズム

correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(t_on_hot,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # 精度

# 学習の実行
sess = tf.Session()

if tf.gfile.Exists('./logdir'):
    tf.gfile.DeleteRecursively('./logdir') # ./logdirが存在する場合削除
writer = tf.summary.FileWriter('./logdir', sess.graph) # 保存先を./logdirに設定

sess.run(tf.global_variables_initializer())
i = 0
for _ in range(N_STEPS):
    cycle = int(N_DATA*3 / BATCH_SIZE)
    begin = int(BATCH_SIZE * (i % cycle))
    end = begin + BATCH_SIZE
    x_batch, t_batch = x_train[begin:end], t_train[begin:end]
    sess.run(train_step, feed_dict={x:x_batch, t:t_batch})
    i += 1
    if i % 10 == 0:
        loss_, acc_ = sess.run([loss, accuracy], feed_dict={x:x_batch,t:t_batch})
        loss_test_, acc_test_ = sess.run([loss, accuracy], feed_dict={x:x_test,t:t_test})
        print("[TRAIN] loss : %f, accuracy : %f" %(loss_, acc_))
        print("[TEST loss : %f, accuracy : %f" %(loss_test_, acc_test_))
sess.close()
