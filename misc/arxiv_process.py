import sys
import codecs
import os
import argparse
import datetime

import glob
import numpy as np
import pickle
from tqdm import tqdm
from collections import Counter

import random
random.seed(123)
np.random.seed(123)

parser = argparse.ArgumentParser(description="preprocessing script for dynamic word embeddings...")
parser.add_argument("--src_dir", type=str, default="/scratch/gjawahar/projects/dynamic_bernoulli_embeddings/dat/arxiv_ML", help="source dir file")
parser.add_argument("--dest_dir", type=str, default="/scratch/gjawahar/projects/objects/d-emb-smart", help="dest dir file")
parser.add_argument("--context_size", type=int, default=2, help="max distance between focus and context word")
parser.add_argument("--dry_run", action='store_true', help="run for a small dataset")
parser.add_argument("--permut", action='store_true', help="randomly shuffle data across time")
args = parser.parse_args()

def gen_windows(numpy_obj):
  num_inputs = (1 + numpy_obj.shape[0] - (2 * args.context_size + 1))
  inputs = []
  for inp in range(num_inputs):
    target = numpy_obj[inp + args.context_size]
    left_context = numpy_obj[inp : inp + args.context_size]
    right_context = numpy_obj[inp+args.context_size+1: inp + (2 * args.context_size + 1)]
    inputs.append([target.item(), np.concatenate([left_context, right_context])])
  return inputs

def process_data(records):
  inputs = []
  for record in records:
    inputs+=gen_windows(np.load(record))
  return inputs, len(inputs)

dat_stats = pickle.load(open(os.path.join(args.src_dir, "dat_stats.pkl"), "rb"), encoding='latin1')

# creating new dir
dest_full_path = args.dest_dir
if not os.path.exists(dest_full_path):
  os.makedirs(dest_full_path)
  os.makedirs(os.path.join(dest_full_path, 'data', 'train'))
  os.makedirs(os.path.join(dest_full_path, 'data', 'dev'))
  os.makedirs(os.path.join(dest_full_path, 'data', 'test'))
print('storing everything in '+dest_full_path)

#copy vocab
#from shutil import copyfile
#copyfile(os.path.join(args.src_dir, "unigram.txt"), os.path.join(dest_full_path, 'data', 'unigram.txt'))

print('loading vocab id maps...')
word2id, id2word, cnt = {}, {}, []
with codecs.open(os.path.join(args.src_dir, "unigram.txt"), 'r', 'utf-8') as f:
  for word_row in f:
    word_str, word_id, count = word_row.strip().split("\t")
    word_id, count = int(word_id), int(count)
    word2id[word_str] = word_id
    id2word[word_id] = word_str
    cnt.append(count)
cnt = np.array(cnt)
unigram_count = cnt

# creating pickle objects
num_tweets_map = [ [-1, -1] for t in dat_stats['T_bins']]
for folder in ['train', 'valid', 'test']:
  print(folder)
  source_folder = os.path.join(args.src_dir, folder)
  master_files = glob.glob(os.path.join(source_folder, "*"))
  for i, t in enumerate(dat_stats['T_bins']):
    dat_files = [f for f in master_files if int(os.path.basename(f)[:dat_stats['prefix']]) == t]
    dat_files = dat_files if args.dry_run==False else [dat_files[0]]
    records, num_words = process_data(dat_files)
    pickle.dump(records, open(os.path.join(dest_full_path, 'data', folder.replace('valid', 'dev'), str(t) + '.pkl' ), 'wb'))
    num_tweets_map[i].append(num_words)

# write dat stats
source_files_names = [[str(t)] for t in dat_stats['T_bins']]
if args.permut == True:
  random.shuffle(source_files_names)
dat_stats = {}
dat_stats['context_size'] = args.context_size
dat_stats['source_files_names'] = source_files_names
dat_stats['num_tweets_map'] = num_tweets_map
dat_stats['unigram_count'] = unigram_count
print(dat_stats)
pickle.dump(dat_stats, open(os.path.join(dest_full_path, 'data', 'dat_stats.pkl'), 'wb'))




