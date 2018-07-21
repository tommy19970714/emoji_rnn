from seq2seq.models import SimpleSeq2Seq
import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import random
import collections
import seq2seq
from seq2seq.models import AttentionSeq2Seq
from seq2seq.models import Seq2Seq
import keras
import util

# シンプルな Seq2Seq モデルを構築
model = SimpleSeq2Seq(input_dim=200, input_length=30, hidden_dim=200, output_length=1, output_dim=1)
#model = AttentionSeq2Seq(input_dim=200, input_length=30, hidden_dim=10, output_length=1, output_dim=342, depth=4)
#model = Seq2Seq(batch_input_shape=(3, 30, 200), hidden_dim=200, output_length=1, output_dim=342, depth=4, peek=True)
# 学習の設定
model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop')

dataloader = util.dataloader()
dataloader.normalize()
X_train, x_test, y_train, y_test = dataloader.dataset() 

# 学習
model.fit(X_train, y_train, nb_epoch=5, batch_size=32, verbose=1)

# 未学習のデータでテスト
#print(model.evaluate(X_test, y_test, batch_size=32))

# 未学習のデータで生成
#predicted = model.predict(X_test, batch_size=32)

#plt.plot()
#plt.savefig('figure.png')
