#!/usr/bin/env python
# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import collections
import numpy as np

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import cross_validation, preprocessing, decomposition #機械学習用のライブラリを利用
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_score
import seaborn as sns

# ベクトルの読み込み
data = pickle.loads(open('./text_vec.pkl', 'rb').read())
label = pickle.loads(open('./label_vec.pkl', 'rb').read())
label_count = len(collections.Counter([str(v) for v in label]))

print("label_count = ", label_count)

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



y = np.array(y)
X = np.array(X).reshape((len(y), 200*30))

print("labelの種類", len(collections.Counter([str(v) for v in y])))

# 主成分分析で次元圧縮
pca = decomposition.PCA(n_components=300)
X_transformed = pca.fit_transform(X)
# 解説5: 主成分分析の結果
print("主成分の分散説明率")
print(pca.explained_variance_ratio_)

# testとtrainの分割
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.20, random_state=random.randint(0, 100))

N_DATA = len(X_train)  # 各クラスの学習用データ数
N_TEST = len(X_test)  # テスト用データ数


# 標準化する
sc = StandardScaler()
sc.fit(X_train)

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# logistic回帰
model = LogisticRegression()
parameters = {'C':[1.0, 10.0, 100.0, 1000.0], 'penalty':['l1', 'l2']}

model2 = GridSearchCV(model, parameters, scoring='accuracy',verbose=3)
model2.fit(X_train_std, y_train)

y_pred = model2.predict(X_test_std)


# 分類結果を表示
print("precision", precision_score(y_test, y_pred, average=None))
print("report", classification_report(y_test, y_pred))
print("accuracy score", accuracy_score(y_test, y_pred))

# seaborn.heatmap を使ってプロットする
conf_mat = confusion_matrix(y_test, y_pred)
index = columns = [str(l) for l in range(label_count)]
df = pd.DataFrame(conf_mat, index=index, columns=columns)

fig = plt.figure(figsize = (7,7))
sns.heatmap(df, annot=False, square=True, fmt='.0f', cmap="Blues")
plt.title('hand_written digit classification')
plt.xlabel('ground_truth')
plt.ylabel('prediction')
fig.savefig("conf_mat.png")
