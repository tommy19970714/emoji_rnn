import MeCab
import db
import pickle
import numpy as np
from gensim.models import word2vec
import re
import numpy as np
import csv
import emoji as emoji_lib

# macabの読み込み
tagger = MeCab.Tagger('-Owakati -d /usr/lib/mecab/dic/mecab-ipadic-neologd')

# ノイズ除去関数
def textNormalize(text):
    text = re.sub(r'https?://[\w/:%#\$&\?\(\)~\.=\+\-…]+', "", text)
    text = re.sub(r"#\w+\b", "", text)# ハッシュタグ
    text = re.sub(r'[!-~]', "", text)  # 半角記号,数字,英字
    text = re.sub(r'[︰-＠]', "", text)  # 全角記号
    text = re.sub("…", "", text)
    return ''.join(text.split())

def remove_emojis(str):
    return ''.join(c for c in str if c not in emoji_lib.UNICODE_EMOJI)

def extract_emojis(str):
  return ''.join(c for c in str if c in emoji_lib.UNICODE_EMOJI)

# emojiの正規表現
emoji_re = re.compile(u'['
 u'\U0001F300-\U0001F64F'
 u'\U0001F680-\U0001F6FF'
 u'\u2600-\u26FF\u2700-\u27BF]', 
 re.UNICODE)

# db load
db_connector = db.DB()
tweets = db_connector.getTweets(offset=0, limit=10000)

# emoji_index load
emoji_index = pickle.loads(open('./emoji_index.pkl', 'rb').read())

# write file open
f = open("./text_label_raw.tsv", "w", newline="")
writer = csv.writer(f, delimiter="\t", quotechar='"', quoting=csv.QUOTE_MINIMAL)


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
            
            # emojiが一つのみのテキストを使用
            match = re.findall(emoji_re, text)
            hit_emoji = ""
            if len(match) == 1:
                if match[0] in emoji_index:
                    #hit_emoji = match[0].encode("utf-8")
                    hit_emoji = emoji_index[match[0]]
                else:
                    continue
            else:
                continue
            no_emoji_text = remove_emojis(text)
            no_emoji_text = re.sub(emoji_re, '', no_emoji_text)
            words = no_emoji_text.split()
            
            writer.writerow([text, hit_emoji])
            print([no_emoji_text, hit_emoji])

    counter += len(tweets)
    tweets = db_connector.getTweets(offset=counter, limit=10000)
    print(counter)
    if counter >= 1000: #1000000:
        break
