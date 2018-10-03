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
import util

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

dataloader = util.dataloader()
dataloader.normalize()
X_train, X_test, y_train, y_test = dataloader.dataset() 


n_hidden = 200
model = RNN(30, n_hidden, dataloader.label_count())

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
        inputs, labels = Variable(inputs.float()), Variable(labels)

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
