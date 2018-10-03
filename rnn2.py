import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import random
import collections
from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding
from keras.layers import LSTM

# ベクトルの読み込み
text_vec = pickle.loads(open('./text_vec.pkl', 'rb').read())
label_vec = pickle.loads(open('./label_vec.pkl', 'rb').read())
label_count = len(collections.Counter([str(v) for v in label_vec]))

# データ作成
X = np.array(text_vec)
y = np.array(label_vec)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=random.randint(0, 100))

# モデルの作成
model = Sequential()
model.add(Embedding(32519, 200, input_length=30))
model.add(LSTM(32))
model.add(Dense(4, activation='sigmoid'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
model.summary()

# 学習
history = model.fit(x_train, y_train, epochs=15, batch_size=32, validation_split=0.2, validation_data=(x_test, y_test))

# 未学習のデータでテスト
#print(model.evaluate(X_test, y_test, batch_size=32))

# 未学習のデータで生成
#predicted = model.predict(X_test, batch_size=32)

#plt.plot()
#plt.savefig('figure.png')
