import MeCab
import db
import pickle
import numpy as np
from gensim.models import word2vec
import re
import numpy as np

# word2vecモデルの読み込み
wordModel = word2vec.Word2Vec.load("../emoji_models/emoji_word2vec_wakati.model")
# macabの読み込み
tagger = MeCab.Tagger('-Owakati -d /usr/lib/mecab/dic/mecab-ipadic-neologd')

# check
print(wordModel["テスト"].shape)
print(type(wordModel["テスト"]))

# ノイズ除去関数
def textNormalize(text):
    text = re.sub(r'https?://[\w/:%#\$&\?\(\)~\.=\+\-…]+', "", text)
    text = re.sub(r"#\w+\b", "", text)# ハッシュタグ
    text = re.sub(r'[!-~]', "", text)  # 半角記号,数字,英字
    text = re.sub(r'[︰-＠]', "", text)  # 全角記号
    text = re.sub("…", "", text)
    return ''.join(text.split())

# db load
db_connector = db.DB()
tweets = db_connector.getTweets(offset=0, limit=10000)

# emoji_index load
emoji_index = pickle.loads(open('./emoji_index.pkl', 'rb').read())

text_vec = []
label_vec = []
counter = 0
while tweets:
    for tweet in tweets:
        for line in tweet["tweet"].splitlines():
            text = textNormalize(line)
            parsed = tagger.parse(text).strip()
            # 短いもの、長いものは除去
            if len(text) < 10:
                continue
            if len(parsed.split()) > 30:
                continue
            
            # emojiの正規表現
            emoji = re.compile(u'['
             u'\U0001F300-\U0001F64F'
             u'\U0001F680-\U0001F6FF'
             u'\u2600-\u26FF\u2700-\u27BF]', 
             re.UNICODE)
            match = re.findall(emoji, line)
            
            # emojiが一つのみのテキストを使用
            hit_emoji = ""
            if len(match) == 1:
                if match[0] in emoji_index:
                    #hit_emoji = match[0].encode("utf-8")
                    hit_emoji = emoji_index[match[0]]
                else:
                    continue
            else:
                continue
            no_emoji_text = re.sub(emoji, '', parsed)
            words = no_emoji_text.split()

            vec = np.zeros((30,200))
            for i,word in enumerate(words):
                if word in wordModel:
                    vec[i] = wordModel[word]
             
            text_vec.append(vec)
            #print(vec)
            label_vec.append(hit_emoji)
            #print(hit_emoji)
    counter += len(tweets)
    tweets = db_connector.getTweets(offset=counter, limit=10000)
    print(counter)
    if counter >= 100000:
        break
open('text_vec.pkl', 'wb').write(pickle.dumps(text_vec) )
open('label_vec.pkl', 'wb').write(pickle.dumps(label_vec) )
