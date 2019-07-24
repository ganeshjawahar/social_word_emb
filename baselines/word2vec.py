import sys
import os
import argparse
import codecs
import gensim
import collections
import random
random.seed(123)
import pickle
import numpy as np
from trainer import trainer
import datetime
from tqdm import tqdm
import glob

parser = argparse.ArgumentParser(description="run baselines for dynamic word embeddings...")

# word2vec params
parser.add_argument('--size', type=int, default=100, help='Dimensionality of the feature vectors.')
parser.add_argument('--sg', type=int, default=1, help=' Defines the training algorithm. If 1, skip-gram is employed; otherwise, CBOW is used.')
parser.add_argument('--window', type=int, default=2, help='The maximum distance between the current and predicted word within a sentence.')
parser.add_argument('--alpha', type=float, default=0.01, help='The initial learning rate.')
parser.add_argument('--min_alpha', type=float, default=0.0001, help='The initial learning rate.')
parser.add_argument('--seed', type=int, default=123, help='Seed for the random number generator.')
parser.add_argument('--min_count', type=int, default=1, help='Ignores all words with total frequency lower than this.')
parser.add_argument('--max_vocab_size', type=int, default=25000, help='Limits the RAM during vocabulary building;')
parser.add_argument('--workers', type=int, default=1, help='Use these many worker threads to train the model.')
parser.add_argument('--hs', type=int, default=0, help='If 1, hierarchical softmax will be used for model training. If set to 0, and negative is non-zero, negative sampling will be used.')
parser.add_argument('--negative', type=int, default=20, help='If > 0, negative sampling will be used, the int for negative specifies how many noise word should be drawn (usually between 5-20). If set to 0, no negative sampling is used.')
parser.add_argument('--iter', type=int, default=10, help='Number of iterations (epochs) over the corpus.')
parser.add_argument('--compute_loss', type=bool, default=True, help='If True, computes and stores loss value which can be retrieved using model.get_latest_training_loss()')
parser.add_argument("--span_unit", type=str, default='yearly', help="monthly or quarterly or halfyearly or yearly")
parser.add_argument("--permut", action='store_true', help="randomly shuffle data across months")

# data params
parser.add_argument('--data', type=str, default='/scratch/gjawahar/projects/data/sosweet_lyon/tokenized/tweet_text/zip_stopwords_oui/splits/full_v2/main', help='data folder')
parser.add_argument('--out', type=str, default='../objects/baselines/word2vec/trial', help='out')

# algorithm
parser.add_argument('--logic', type=int, default=0, help='0=vanilla; 1=kim; 2=hamilton;')
parser.add_argument('--threshold', type=float, default=0.999, help='for kims method, threshold to stop training')

args =  parser.parse_args()

def getFilesByUnit(cand_files_names):
  source_files_names = []
  if args.span_unit == 'monthly':
    for fname in cand_files_names:
      source_files_names.append([fname])
  elif args.span_unit == 'halfyearly':
    cur_items = []
    for fname in cand_files_names:
      year = int(fname.split("-")[0])
      if year == 2014:
        cur_items.append(fname)
    source_files_names.append(cur_items)
    cur_items = []
    for fname in cand_files_names:
      month = int(fname.split("-")[1])
      year = int(fname.split("-")[0])
      if year != 2014:
        if month==6 or month==12:
          cur_items.append(fname)
          source_files_names.append(cur_items)
          cur_items = []
        else:
          cur_items.append(fname)
    if len(cur_items)>0:
      source_files_names.append(cur_items)
  elif args.span_unit == 'yearly':
    cur_year, cur_items = 2014, []
    for fname in cand_files_names:
      year = int(fname.split("-")[0])
      if year == cur_year:
        cur_items.append(fname)
      else:
        source_files_names.append(cur_items)
        cur_items = [fname]
        cur_year = year
    if len(cur_items)>0:
      source_files_names.append(cur_items)
  elif args.span_unit == 'quarterly':
    for i in range(len(cand_files_names)//3):
      source_files_names.append([cand_files_names[(3*i)], cand_files_names[(3*i)+1], cand_files_names[(3*i)+2]])
  return source_files_names

source_files_abs = glob.glob(os.path.join(args.data, 'train', '*'))
cand_files_names = []
for file in source_files_abs:
  cand_files_names.append(file.split("/")[-1])
cand_files_names = sorted(cand_files_names, key=lambda x: datetime.datetime.strptime(x, '%Y-%m'))
source_files_names = getFilesByUnit(cand_files_names)
if args.permut == True:
  random.shuffle(source_files_names)
#source_files_names = source_files_names[1:4] + source_files_names[len(source_files_names)-3:]
print('source files (in order)...')
print(source_files_names)

print('counting corpus size...')
wordCounter = collections.Counter()
corpusSize = []
for source_f in tqdm(source_files_names):
  count = 0
  for file in source_f:
    with codecs.open(os.path.join(args.data, 'train', file) , 'r', 'utf-8') as f:
      for line in f:
        # content = line.split('\t')[1].split()
        count = count + 1
        break
  corpusSize.append(count)

print('loading vocab...')
vocab = {}
word2id, id2word = {}, {}
with codecs.open(os.path.join(args.data, 'vocab_25K'), 'r', 'utf-8') as f:
  for word_row in f:
    word_id, word_str, count = word_row.strip().split("\t")
    word_id, count = int(word_id), int(count)
    word2id[word_str] = word_id
    id2word[word_id] = word_str
    vocab[word_str] = True

# creating dirs to save the model
if not os.path.isdir(args.out):
  os.makedirs(args.out)

# training
print('training started...')
targets, contexts = trainer(args, word2id, source_files_names, corpusSize, args.out)

# saving
print('saving objects...')
full_out_name = os.path.join(args.out, 'stats.pkl')
save_obj = {'time':source_files_names, 'context_size': args.window, 'word_dim': args.size, 'targets':targets, 'contexts':contexts}
pickle.dump(save_obj, open(full_out_name, 'wb'))


