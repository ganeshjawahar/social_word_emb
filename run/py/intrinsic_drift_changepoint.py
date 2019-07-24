# Finding changing words with absolute drift and
# changepoint analysis
# ref: http://www.cs.columbia.edu/~blei/papers/RudolphBlei2018.pdf

import sys
import os
import argparse
import pickle
import codecs
import numpy as np
import glob
import operator
from numpy import dot
from numpy.linalg import norm

parser = argparse.ArgumentParser(description="evaluation script for finding changing words with absolute drift and changepoint analysis")
parser.add_argument("--vocab_file", type=str, default="/Users/ganeshj/Desktop/dw2v/naacl19/vocab_25K", help="path to the file containing the vocabulary words")
parser.add_argument("--embed_dir", type=str, default="/home/ganesh/objects/dywoev/trial/result/dyntrain/run1", help="directory containing dynamic word embeddings to be evaluated")
parser.add_argument('--K', type=int, default=10, help='top <int> words to be shown')
parser.add_argument("--sim_met", type=str, default="cosine", help="similarity metric to be used? cosine or euclidean")
args = parser.parse_args()
print(args)

# load vocabulary
word2id, id2word = {}, {}
with codecs.open(args.vocab_file, 'r', 'utf-8') as f:
  for word_row in f:
    word_id, word_str, count = word_row.strip().split("\t")
    word_id, count = int(word_id), int(count)
    word2id[word_str] = word_id
    id2word[word_id] = word_str

# load word embeddings
is_baseline = os.path.exists(os.path.join(args.embed_dir, 'stats.pkl'))
if is_baseline:
  train_embed = pickle.load(open(os.path.join(args.embed_dir, 'stats.pkl'), 'rb'))
  time_period = train_embed["time"]
  context_size = train_embed["context_size"]
  word_dim = train_embed["word_dim"]
  context_embeds, target_embeds, mean_context_embeds, mean_target_embeds = {}, {}, {}, {}
  for cur_time, (context_embed, target_embed) in enumerate(zip(train_embed["contexts"], train_embed["targets"])):
    context_embeds[cur_time] = context_embed
    target_embeds[cur_time] = target_embed
    mean_context_embed, mean_target_embed, num_entries = np.zeros(word_dim, dtype=np.float32), np.zeros(word_dim, dtype=np.float32), 0.0
    for word in context_embed.keys():
      if word in word2id:
        mean_target_embed += target_embed[word]
        mean_context_embed += context_embed[word]
        num_entries += 1.0
    mean_target_embed /= num_entries
    mean_context_embed /= num_entries
    mean_target_embeds[cur_time] = mean_target_embed
    mean_context_embeds[cur_time] = mean_context_embed
else:
  model_id = max([ int(file.split('/')[-1].split('_')[-1].split('.')[0]) for file in glob.glob(os.path.join(args.embed_dir, 'embed_*.pkl'))])
  model_file = os.path.join(args.embed_dir, 'embed_%d.pkl'%model_id)
  print("loading model %s"%model_file)
  target_embeds = pickle.load(open(model_file, 'rb'))

# find words with max. absolute drift
def iswordpresentinallperiods(word_str, embeds):
  for i in range(len(embeds)):
    if word_str not in embeds[i]:
      return False
  return True
def similarity(a, b):
  if args.sim_met == 'cosine':
    return dot(a, b)/(norm(a)*norm(b))
  return np.linalg.norm(a-b)

'''
query_words = ['bataclan', 'trump', 'syria', 'bachar']
def find_neighbors(word, embeds):
  targ_wvec = embeds[word] if is_baseline else embeds[word2id[word]]
  word2sim = {}
  for word_str in word2id:
    if word_str != word:
      if is_baseline and word_str in embeds:
        word2sim[word_str] = similarity(targ_wvec, embeds[word_str])
      elif not is_baseline:
        word2sim[word_str] = similarity(targ_wvec, embeds[word2id[word_str]])
  word2sim = sorted(word2drift.items(), key=operator.itemgetter(1), reverse=True)[0:args.K]
  res = word + '\n'
  for word_sim in word2sim:
    n_word, sim = word_sim
    res += n_word + '\n'
  return res + "\n"
for qword in query_words:
  if qword in word2id:
    if is_baseline:
      if qword in target_embeds[0]:
        print(find_neighbors(qword, target_embeds[0]))
    else:
      print(find_neighbors(qword, target_embeds[0]))
sys.exit(0)
'''

word2drift = {}
for word_str in word2id:
  word_id = word2id[word_str]
  if is_baseline:
    #if iswordpresentinallperiods(word_str, target_embeds):
    if word_str in target_embeds[4] and word_str in target_embeds[0]:
      word2drift[word_str] = similarity(target_embeds[4][word_str], target_embeds[0][word_str])
  else:
    word2drift[word_str] = similarity(target_embeds['target_4'][word_id], target_embeds['target_0'][word_id])
word2drift = sorted(word2drift.items(), key=operator.itemgetter(1), reverse=True)
word2drift = word2drift[0:args.K]
print(word2drift)

# find changepoints
norm_term = []
for t in range(4):
  cur_val = 0.0
  for word_str in word2id:
    word_id = word2id[word_str]
    if is_baseline:
      if word_str in target_embeds[t] and word_str in target_embeds[t+1]:
        cur_val += similarity(target_embeds[t][word_str], target_embeds[t+1][word_str])
    else:
      cur_val += similarity(target_embeds['target_%d'%t][word_id], target_embeds['target_%d'%(t+1)][word_id])
  norm_term.append(cur_val)
word2drift = ['bataclan', 'paris', 'trump', 'emmanuel', 'euro2016', 'equipedefrance']
for word in word2drift:
  word_str = word#[0]
  period2change = {}
  for t in range(4):
    period = '%d=>%d'%(2014+t, 2014+t+1)
    if is_baseline:
      if word_str in target_embeds[t] and word_str in target_embeds[t+1]:
        change = similarity(target_embeds[t][word_str], target_embeds[(t+1)][word_str])/norm_term[t]
        period2change[period] = change
    else:
      change = similarity(target_embeds['target_%d'%t][word_id], target_embeds['target_%d'%(t+1)][word_id])/norm_term[t]
      period2change[period] = change
  period2change = sorted(period2change.items(), key=operator.itemgetter(1), reverse=True)
  if len(period2change) > 0:
    res = '%s | %s'%(word_str, period2change[0][0])
    if len(period2change) > 1:
      res += ' %s'%period2change[1][0]
    print(res)














