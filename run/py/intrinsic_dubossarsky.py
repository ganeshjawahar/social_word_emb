# compute the change in the meanings of words correlated with distance from centroid for different clusters
# http://ceur-ws.org/Vol-1347/paper14.pdf

import sys
import os
import argparse
import pickle
import codecs
import glob
from sklearn import metrics
from spherecluster import SphericalKMeans
import numpy as np
from numpy import dot
from numpy.linalg import norm
from sklearn.cluster import KMeans
from scipy.stats.stats import pearsonr

parser = argparse.ArgumentParser(description="evaluation script for getting clustering scores")
parser.add_argument("--data_dir", type=str, default="/home/ganesh/data/sosweet/full", help="directory containing labels")
parser.add_argument("--embed_dir", type=str, default="/home/ganesh/objects/dywoev/sept28_meta_search/user_category_dept_feat/result/dyntrain/sample_run", help="directory containing dynamic word embeddings to be evaluated")
args = parser.parse_args()
print(args)

# load vocabulary
word2id, id2word = {}, {}
with codecs.open(os.path.join(args.data_dir, 'main', 'vocab_25K'), 'r', 'utf-8') as f:
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

interesting_words = {}
for idi in range(7000):
  word = id2word[idi]
  interesting_words[word] = True

def similarity(a, b):
  return 1 - dot(a, b)/(norm(a)*norm(b))
def justcos(a, b):
  return dot(a, b)/(norm(a)*norm(b))
def prototype_centers(cluster_centers, labels, embeds):
  proto_center = {}
  for i in range(len(labels)):
    label = labels[i]
    if label not in proto_center:
      proto_center[label] = embeds[i]
    else:
      if justcos(embeds[i], cluster_centers[label]) > justcos(proto_center[label], cluster_centers[label]):
        proto_center[label] = embeds[i]
  return proto_center
def vectorize(mp):
  arr = []
  for word in interesting_words:
    arr.append(mp[word])
  return np.array(arr, dtype=np.float32)

for cluster_k in [100, 250, 500, 750, 1000]:
  word2change = {}
  for word in interesting_words:
    wid = word2id[word]
    word_vec_0 = target_embeds['target_0'][wid]
    word_vec_4 = target_embeds['target_4'][wid]
    word2change[word] = similarity(word_vec_4, word_vec_0)
  word2distance_0_actual, word2distance_0_proto = {}, {}
  kmeans = KMeans(n_clusters=cluster_k, random_state=0, n_jobs=1).fit(target_embeds['target_0'][0:len(word2id)])
  labels, cluster_centers = kmeans.labels_, kmeans.cluster_centers_
  proto_centers = prototype_centers(cluster_centers, labels, target_embeds['target_0'][0:len(word2id)])
  for word in interesting_words:
    wid = word2id[word]
    word_vec_0 = target_embeds['target_0'][wid]
    word_vec_cluster_center = cluster_centers[labels[wid]]
    word2distance_0_actual[word] = similarity(word_vec_cluster_center, word_vec_0)
    word2distance_0_proto[word] = similarity(proto_centers[labels[wid]], word_vec_0)
  word2distance_4_actual, word2distance_4_proto = {}, {}
  kmeans = KMeans(n_clusters=cluster_k, random_state=0, n_jobs=1).fit(target_embeds['target_4'][0:len(word2id)])
  labels, cluster_centers = kmeans.labels_, kmeans.cluster_centers_
  proto_centers = prototype_centers(cluster_centers, labels, target_embeds['target_4'][0:len(word2id)])
  for word in interesting_words:
    wid = word2id[word]
    word_vec_4 = target_embeds['target_4'][wid]
    word_vec_cluster_center = cluster_centers[labels[wid]]
    word2distance_4_actual[word] = similarity(word_vec_cluster_center, word_vec_4)
    word2distance_4_proto[word] = similarity(proto_centers[labels[wid]], word_vec_4)
  word2change = vectorize(word2change)
  word2distance_0_actual = vectorize(word2distance_0_actual)
  word2distance_0_proto = vectorize(word2distance_0_proto)
  word2distance_4_actual = vectorize(word2distance_4_actual)
  word2distance_4_proto = vectorize(word2distance_4_proto)
  actual_correlation = (pearsonr(word2change, word2distance_0_actual)[0] + pearsonr(word2change, word2distance_4_actual)[0])/2
  proto_correlation = (pearsonr(word2change, word2distance_0_proto)[0] + pearsonr(word2change, word2distance_4_proto)[0])/2
  print('K=%d; actual=%.5f; proto=%.5f;'%(cluster_k, actual_correlation, proto_correlation))


















