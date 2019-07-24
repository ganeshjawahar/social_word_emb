# for every meta feature, find the most attentive words

import sys
import os
import numpy as np
import pickle
import codecs

vocab_path = '/home/ganesh/data/sosweet/full/main/vocab_25K'
attn_pkl_path = '/home/ganesh/objects/dywoev/plotattn/result/dyntrain/context/attn.pkl'
maxhits = 5
paths = 'content/all_tweets_50.txt,content/geo_tweets_50.txt,dept/feat.txt,income/insee_feat.txt,income/iris_feat.txt,network/feat.txt,presence/feat.txt,region/feat.txt,yt_category/feat.txt'

# read the vocabulary
id2word = {}
with codecs.open(vocab_path, 'r', 'utf-8') as f:
  for line in f:
    content = line.strip().split("\t")
    id2word[int(content[0])] = content[1]
print(len(id2word))

# read the attn file
attn_pkl = pickle.load(open(attn_pkl_path, 'rb')) 
def sortWords(attn_mat):
  res = []
  for fi in range(attn_mat.shape[1]):
    x = np.argsort(attn_mat[:,fi], axis=0)
    x = x[0:maxhits]
    x = np.flipud(x)
    for xi in range(maxhits):
      if len(res) < xi+1:
        res.append([])
      res[xi].append(id2word[x[xi]])
  print(paths)
  for r in res:
    print(','.join(r))

sortWords(attn_pkl[2])



