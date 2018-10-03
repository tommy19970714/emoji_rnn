#!/usr/bin/env python
# coding: utf-8
import random
import collections
import numpy as np

import pickle
import pandas as pd
from sklearn.model_selection import train_test_split

class dataloader:
    def __init__(self, dataPath='./text_vec.pkl', labelPath='./label_vec.pkl'):
        # ベクトルの読み込み
        self.data = pickle.loads(open(dataPath, 'rb').read())
        self.label = pickle.loads(open(labelPath, 'rb').read())
        # データ作成
        self.X = []
        self.y = []

    def normalize(self):
        # データ数を合わせる
        c = collections.Counter(self.label)
        sample_nums = c.most_common()
        print("sample_nums:", sample_nums)

        min_num = np.min([s[1] for s in sample_nums])
        print("min_num:", min_num)

        for sample in sample_nums:
            diff_num = int(sample[1] - min_num)
            print("クラス%d 削除サンプル数: %d (%0.2f％)" % (sample[0], diff_num, (diff_num/sample[1])*100))
            indexes = [i for i, l in enumerate(self.label) if l == sample[0]]
            del_indexes = random.sample(indexes, min_num)
            self.X.extend([self.data[i] for i in indexes])
            self.y.extend([self.label[i] for i in indexes])
        self.X = np.array(self.X)
        self.y = np.array(self.y)

   
    def dataset(self):
        return train_test_split(self.X, self.y, test_size=0.20, random_state=random.randint(0, 100))

    def label_count(self):
        return len(collections.Counter([str(v) for v in self.label]))
    
    def rawdata(self):
        return self.X, self.y
