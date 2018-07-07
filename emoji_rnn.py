from seq2seq.models import SimpleSeq2Seq
import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import random

# ベクトルの読み込み
text_vec = pickle.loads(open('./text_vec.pkl', 'rb').read())
label_vec = pickle.loads(open('./label_vec.pkl', 'rb').read())

# シンプルな Seq2Seq モデルを構築
model = SimpleSeq2Seq(input_dim=200, input_length=30, hidden_dim=10, output_length=1, output_dim=1)

# 学習の設定
model.compile(loss='mse', optimizer='rmsprop')

# データ作成
X = np.array(text_vec)
y = np.array(label_vec)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=random.randint(0, 100))

# 学習
model.fit(X_train, y_train, nb_epoch=5, batch_size=32)

# 未学習のデータでテスト
print(model.evaluate(X_test, y_test, batch_size=32))

# 未学習のデータで生成
predicted = model.predict(X_test, batch_size=32)

#plt.plot()
#plt.savefig('figure.png')
