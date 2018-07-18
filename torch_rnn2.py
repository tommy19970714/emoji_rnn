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

from torch.autograd import Variable #自動微分用
import torch.nn as nn #ネットワーク構築用
import torch.optim as optim #最適化関数
import torch.nn.functional as F #ネットワーク用の様々な関数
import torch.utils.data #データセット読み込み関連
import torchvision #画像関連

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
 
        self.hidden_size = hidden_size
 
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
 
    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden
 
    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

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


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=random.randint(0, 100))


n_hidden = 128
model = RNN(30, n_hidden, label_count)

train = torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
train_loader = torch.utils.data.DataLoader(train, batch_size=100, shuffle=True)
test = torch.utils.data.TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
test_loader = torch.utils.data.DataLoader(test, batch_size=100, shuffle=True)


criterion = nn.NLLLoss()

learning_rate = 0.005
hidden = model.initHidden()
#トレーニング
#エポック数の指定
for epoch in range(2):  # loop over the dataset multiple times
    #データ全てのトータルロス
    running_loss = 0.0 

    for i, data in enumerate(train_loader):
        #入力データ・ラベルに分割
        # get the inputs
        inputs, labels = data

        # Variableに変形
        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)

        # optimizerの初期化
        # zero the parameter gradients
        model.zero_grad()

        output, hidden = model(inputs, hidden)

        loss = criterion(output, input)
        loss.backward()

        running_loss += loss.data[0]
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
