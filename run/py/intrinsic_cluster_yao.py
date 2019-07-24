# perform word clustering evaluation
# based on https://arxiv.org/pdf/1703.00607.pdf

import sys
import os
import argparse
import pickle
import codecs
import glob
from sklearn import metrics
from spherecluster import SphericalKMeans
import numpy as np

parser = argparse.ArgumentParser(description="evaluation script for getting clustering scores")
parser.add_argument("--label_dir", type=str, default="/home/ganesh/data/sosweet/full", help="directory containing labels")
parser.add_argument("--embed_dir", type=str, default="/home/ganesh/objects/dywoev/trial/result/dyntrain/run1", help="directory containing dynamic word embeddings to be evaluated")
parser.add_argument("--handle_missing", action='store_false', help="substitute the missing word embeddings with the mean value. works for baseline embeddings.")
args = parser.parse_args()
print(args)

# load vocabulary
word2id, id2word = {}, {}
with codecs.open(os.path.join(args.label_dir, 'main', 'vocab_25K'), 'r', 'utf-8') as f:
  for word_row in f:
    word_id, word_str, count = word_row.strip().split("\t")
    word_id, count = int(word_id), int(count)
    word2id[word_str] = word_id
    id2word[word_id] = word_str

# load labels
triplets = []
year2id = {'2014':0, '2015':1, '2016':2, '2017':3, '2018':4}
cat2id, id2cat = {}, {}
with codecs.open(os.path.join(args.label_dir, 'intrinsic', 'yao_clustering', 'data.txt'), 'r', 'utf-8') as f:
  for line in f:
    word, year, cat, _ = line.strip().split("\t")
    if cat not in cat2id:
      cat2id[cat] = len(cat2id)
      id2cat[cat2id[cat]] = cat
    triplets.append([word2id[word], year2id[year], cat2id[cat]])
print('# unique categories = %d'%len(id2cat))

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
    mean_context_embed, mean_target_embed, num_entries = np.zeros(word_dim), np.zeros(word_dim), 0.0
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
  if 'target_0' in target_embeds:
    word_dim = target_embeds['target_0'].shape[1]
  else:
    word_dim = target_embeds['target'].shape[1]
    target_embed = target_embeds['target']
    target_embeds = {}
    for yi in range(5):
      target_embeds['target_%d'%yi] = target_embed

# prepare X
X, Y_true = [], []
for item in triplets:
  word, year, cat = item
  if is_baseline:
    if id2word[word] in target_embeds[year]:
      X.append(target_embeds[year][id2word[word]])
      Y_true.append(cat)
    elif args.handle_missing:
      X.append(mean_target_embeds[year])
      Y_true.append(cat)
  else:
    embed_mat = target_embeds['target_%d'%year]
    X.append(embed_mat[word])
    Y_true.append(cat)

X = np.array(X, dtype=np.float32)
print(X.shape)
print('# <word,year> tuples for which dynamic word embedding is not found = %d'%(len(triplets)-X.shape[0]))

# perform clustering
skm = SphericalKMeans(n_clusters=len(cat2id))
skm.fit(X)
Y_pred = skm.labels_
print("NMI = %.4f"%(metrics.normalized_mutual_info_score(Y_true, Y_pred)))



