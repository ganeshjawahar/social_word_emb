# Evaluation based on synthetic injection of linguistic change
# ref: http://www.perozzi.net/publications/15_www_linguistic.pdf

import sys
import os
import argparse
import pickle
import codecs
import numpy as np
import glob
from numpy import dot
from numpy.linalg import norm
import math
import operator
from tqdm import tqdm

'''
# code to convert stats file to be picklable in python 2.7
for source_folder in tqdm(glob.glob("/home/ganesh/objects/dywoev/baselines/word2vec/*")):
  if "_" in source_folder.split("/")[-1] and not os.path.exists(os.path.join(source_folder, 'stats.pkl')):
    continue
  if not source_folder.split("/")[-1].startswith("kim_syn_"):
    continue
  pkl = pickle.load(open(os.path.join(source_folder, 'stats.pkl'), 'rb'))
  pickle.dump(pkl, open(os.path.join(source_folder, 'stats_python27.pkl'), 'wb'), protocol=2)
  print("created %s"%(os.path.join(source_folder, 'stats_python27.pkl')))
sys.exit(0)
'''

'''
# code to create script that runs across types: {freq, synt}, repl_prob and baselines
baselines = ['vanilla', 'kim', 'vanilla']
repl_prb = [0.2, 0.4, 0.6, 0.8]
types = ['frequent', 'syntactic']
for baseline in baselines:
  for typ in types:
    for rp in repl_prb:
      embed_dir = "/home/ganesh/objects/dywoev/baselines/word2vec/%s_%s_%s"%(baseline, typ[0:3], str(rp).replace(".", "p"))
      perturb_folder = "%s/prob_%.1f"%(typ, rp)
      res = 'python run/py/intrinsic_kulkarni_synthetic.py --embed_dir %s --perturb_folder %s'%(embed_dir, perturb_folder)
      print(res)
sys.exit(0)
'''

from changepoint.mean_shift_model import MeanShiftModel
parser = argparse.ArgumentParser(description="evaluation script for synthetic injection of linguistic change")
parser.add_argument("--data_dir", type=str, default="/home/ganesh/data/sosweet/full", help="directory containing data")
parser.add_argument("--embed_dir", type=str, default="/home/ganesh/objects/dywoev/baselines/word2vec/vanilla_fre_0p8", help="directory containing dynamic word embeddings to be evaluated")
parser.add_argument("--perturb_folder", type=str, default="frequent/prob_0.8", help="type of perturbation: frequent or syntactic? replacement probability: 0.2 or 0.4 or 0.6 or 0.8?")
parser.add_argument('--num_periods', type=int, default=6, help='number of time slices involved')
args = parser.parse_args()
print(args)

# load vocabulary
word2id, id2word = {}, {}
with codecs.open(os.path.join(args.data_dir, 'intrinsic', 'kulkarni_eval', args.perturb_folder, 'vocab_25K'), 'r', 'utf-8') as f:
  for word_row in f:
    word_id, word_str, count = word_row.strip().split("\t")
    word_id, count = int(word_id), int(count)
    word2id[word_str] = word_id
    id2word[word_id] = word_str

# load word embeddings
is_baseline = os.path.exists(os.path.join(args.embed_dir, 'stats.pkl'))
if is_baseline:
  train_embed = pickle.load(open(os.path.join(args.embed_dir, 'stats_python27.pkl'), 'rb'))
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
  model_files = glob.glob(os.path.join(args.embed_dir, 'embed_*_py27.pkl'))
  model_file = model_files[0] #os.path.join(args.embed_dir, 'embed_%d_py27.pkl'%model_id)
  print("loading model %s"%model_file)
  target_embeds = pickle.load(open(model_file, 'rb'))
  word_dim = target_embeds['target_0'].shape[1]

# read donor => receptor mapping
pertubed_words = {}
with codecs.open(os.path.join(args.data_dir, 'intrinsic', 'kulkarni_eval', args.perturb_folder, 'donor2receptor'), 'r', 'utf-8') as f:
  for line in f:
    donor, receptor = line.strip().split("\t")
    pertubed_words[donor] = True
    pertubed_words[receptor] = True
print('# pertubed_words = %d'%len(pertubed_words))

# find words present in all time slices
common_words = {}
if is_baseline:
  def is_common_word(word):
    for t in range(len(target_embeds)):
      if word not in target_embeds[t]:
        return False
    return True
  for word in word2id:
    if is_common_word(word):
      common_words[word] = True
else:
  for word in word2id:
    common_words[word] = True
print('# common words = %d/%d (%d)'%(len(common_words), len(word2id), (100.0*len(common_words))/len(word2id) ))

# construct time series
def cosine_similarity(a, b):
  return dot(a, b)/(norm(a)*norm(b))
def get_wv(word_str, year_id):
  if is_baseline:
    return target_embeds[year_id][word_str]
  return target_embeds['target_%d'%year_id][word2id[word_str]]
word2timeseries = {}
for word_str in common_words:
  series = []
  for t in range(args.num_periods-1):
    series.append(1.0 - cosine_similarity(get_wv(word_str, t+1), get_wv(word_str, 0)))
  word2timeseries[word_str] = series


# construct normalized timeseries
# find mean, variance
mean_vec = np.zeros(args.num_periods, dtype=np.float32)
for word in word2timeseries:
  for ti, time in enumerate(word2timeseries[word]):
    mean_vec[ti] += time
mean_vec /= len(word2timeseries)
var_vec = np.zeros(args.num_periods, dtype=np.float32)
for word in word2timeseries:
  for ti, time in enumerate(word2timeseries[word]):
    var_vec[ti] += (time - mean_vec[ti])**2
var_vec /= len(word2timeseries)
word2normalizedtimeseries = {}
for word in word2timeseries:
  un_norm_series = word2timeseries[word]
  norm_series = []
  for i, val in enumerate(un_norm_series):
    norm_score = (val - mean_vec[i])/math.sqrt(var_vec[i])
    if norm_score>1.75:
      norm_series.append(norm_score)
  word2normalizedtimeseries[word] = norm_series
word2timeseries = word2normalizedtimeseries  

# use mean shift models to extract p-values
word2pval = {}
for word in tqdm(word2id):
  if word not in word2timeseries or len(word2timeseries[word])<2:
    continue
  model = MeanShiftModel()
  stats_ts, pvals, nums = model.detect_mean_shift(word2timeseries[word], B=1000)
  word2pval[word] = np.array(pvals, dtype=np.float32).min()

# compute mrr
sorted_word2pval = sorted(word2pval.items(), key=operator.itemgetter(1))
mrr = 0.0
for i, tup in enumerate(sorted_word2pval):
  word, pval = tup[0], tup[1]
  if word in pertubed_words:
    mrr += 1.0/(i+1.0)
print('mrr = %.10f'%(mrr/len(pertubed_words)))



