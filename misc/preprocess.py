'''
preprocessing code for d-emb
'''

import sys
import codecs
import os
import argparse
import datetime

import glob
import numpy as np
import _pickle as pickle
from tqdm import tqdm
from collections import Counter

import random
random.seed(123)
np.random.seed(123)

parser = argparse.ArgumentParser(description="preprocessing script for dynamic word embeddings...")
parser.add_argument("--src_dir", type=str, default="/home/ganesh/data/sosweet/full/main", help="source dir file")
parser.add_argument("--permut", action='store_true', help="randomly shuffle data across time")
parser.add_argument("--dest_dir", type=str, default="/home/ganesh/objects/dywoev/monthly", help="dest dir file")
parser.add_argument("--context_size", type=int, default=2, help="max distance between focal and context word")
parser.add_argument("--dry_run", action='store_true', help="run for a small dataset")
parser.add_argument("--tabbed_data", action='store_false', help="each line has <something><tab><tweet>??")
parser.add_argument("--within_tweet", action='store_false', help="consider context within tweet only?")
parser.add_argument("--span_unit", type=str, default='yearly', help="monthly or quarterly or halfyearly or yearly")
args = parser.parse_args()

'''
utility functions
'''
def tokenize(line):
  if args.tabbed_data == True:
    return line.strip().split("\t")[-1].split()
  return line.strip().split()

def gen_windows(numpy_obj):
  num_inputs = (1 + numpy_obj.shape[0] - (2 * args.context_size + 1))
  inputs = []
  for inp in range(num_inputs):
    target = numpy_obj[inp + args.context_size]
    left_context = numpy_obj[inp : inp + args.context_size]
    right_context = numpy_obj[inp+args.context_size+1: inp + (2 * args.context_size + 1)]
    inputs.append([target.item(), np.concatenate([left_context, right_context])])
  return inputs

def padRecord(rec_np):
  items = []
  
  # left pad items
  #items += [len(word2id)+i for i in range(args.context_size)]
  items += [len(word2id) for i in range(args.context_size)]

  # actual tweet
  items += rec_np.tolist()

  # right pad items
  #items += [len(word2id)+i+args.context_size for i in range(args.context_size)]
  items += [len(word2id) for i in range(args.context_size)]

  return np.array(items)

def process_data(records, pad=False):
  inputs = []
  for record in records:
    if pad:
      record = padRecord(record)
    inputs+=gen_windows(record)
  return inputs, len(inputs)

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

# creating new dir
#dest_folder_name = datetime.datetime.now().strftime('%d_%m_%Y_%H_%M_%S')
dest_full_path = args.dest_dir
if not os.path.exists(dest_full_path):
  os.makedirs(dest_full_path)
if not os.path.exists(os.path.join(dest_full_path, 'data', 'train')):
  os.makedirs(os.path.join(dest_full_path, 'data', 'train'))
if not os.path.exists(os.path.join(dest_full_path, 'data', 'dev')):
  os.makedirs(os.path.join(dest_full_path, 'data', 'dev'))
if not os.path.exists(os.path.join(dest_full_path, 'data', 'test')):
  os.makedirs(os.path.join(dest_full_path, 'data', 'test'))
print('storing everything in '+dest_full_path)

# get all data files
source_files_abs = glob.glob(os.path.join(args.src_dir, 'train', '*'))
cand_files_names = []
for file in source_files_abs:
  cand_files_names.append(file.split("/")[-1])
cand_files_names = sorted(cand_files_names, key=lambda x: datetime.datetime.strptime(x, '%Y-%m'))
source_files_names = getFilesByUnit(cand_files_names)
if args.permut == True:
  random.shuffle(source_files_names)
if len(source_files_names)>5:
  source_files_names = source_files_names[1:4] + source_files_names[len(source_files_names)-3:]
print('source files (in order)...')
print(source_files_names)

print('loading vocab id maps...')
word2id, id2word, cnt = {}, {}, []
with codecs.open(os.path.join(args.src_dir, 'vocab_25K'), 'r', 'utf-8') as f:
  for word_row in f:
    word_id, word_str, count = word_row.strip().split("\t")
    word_id, count = int(word_id), int(count)
    word2id[word_str] = word_id
    id2word[word_id] = word_str
    cnt.append(count)
cnt = np.array(cnt)
unigram_count = cnt
cnt = 1.0*cnt/cnt.sum()

#print('saving vocab...')
#vocab_fil_w = codecs.open(os.path.join(dest_full_path, 'data', 'unigram.txt'), 'w', 'utf-8')
#for word_count in vocab_words:
#  vocab_fil_w.write(word_count[0]+'\t'+str(word2id[word_count[0]])+'\t'+str(word_count[1])+'\n')
#vocab_fil_w.close()

print('creating pickle objects...')
pbar = tqdm(total=len(cand_files_names))
total_tweets_processed, total_tweets_removed = 0, 0
num_tweets_map = []
for source_file_name_arr in source_files_names:
  cur_tweets_processed, cur_tweets_removed, num_split = 0, 0, [0]*3
  for source_file_name in source_file_name_arr:
    pbar.update(1)
    cand_folders = ['train', 'dev', 'test']
    cand_size = [3000, 2000, 2000]
    num_words_master = []
    within_tweet_vals = [args.within_tweet, True, True]
    ci = 0
    for cand_folder, within_tweet_val in zip(cand_folders, within_tweet_vals):
      source_file_abs = os.path.join(args.src_dir, cand_folder, source_file_name)
      if not os.path.exists(source_file_abs):
        continue
      input_plain = []
      cand_tweets_processed = 0
      with codecs.open(source_file_abs, 'r', 'utf-8', errors='ignore') as file:
        for line in file:
          if args.dry_run == True and cand_tweets_processed==cand_size[ci]:
            break
        
          cur_words = tokenize(line)
          cur_vocab_words = []
          for word in cur_words:
            if word in word2id:
              cur_vocab_words.append(word)
          
          total_tweets_processed+=1
          
          cur_tweet = []
          for word in cur_vocab_words:
            cur_tweet.append(word2id[word])

          if len(cur_tweet)==0:
            total_tweets_removed+=1
            cur_tweets_removed+=1
            continue

          if within_tweet_val == True:
            cur_tweet = np.array(cur_tweet)
            
            '''
            if cur_tweet.shape[0] < (2*args.context_size + 1):
              total_tweets_removed+=1
              cur_tweets_removed+=1
              continue
            '''
            if ci == 0:
              prob = np.random.uniform(0,1,cur_tweet.shape)
              p = 1 - np.sqrt((10.0**(-5))/cnt[cur_tweet])
              cur_tweet = cur_tweet[prob > p] 
            
            '''
            if cur_tweet.shape[0] < (2*args.context_size + 1):
              total_tweets_removed+=1
              cur_tweets_removed+=1
              continue
            '''
            input_plain.append(cur_tweet)
          else:
            input_plain+=cur_tweet

          cand_tweets_processed+=1
      cur_tweets_processed+=cand_tweets_processed
      if within_tweet_val == True:
        records, num_words = process_data(input_plain, pad=True)
      else:
        input_plain = np.array(input_plain)
        prob = np.random.uniform(0,1,input_plain.shape)
        p = 1 - np.sqrt((10.0**(-5))/cnt[input_plain])
        input_plain = input_plain[prob > p]
        records, num_words = process_data([input_plain])
      pickle.dump(records, open(os.path.join(dest_full_path, 'data', cand_folder, source_file_name + '.pkl' ), 'wb'))
      num_split[ci]+=num_words
      ci = ci + 1
  num_tweets_map.append([cur_tweets_processed, cur_tweets_removed] + num_split)
pbar.close()

# write dat stats
dat_stats = {}
dat_stats['permut'] = args.permut
dat_stats['within_tweet'] = args.within_tweet
dat_stats['context_size'] = args.context_size
dat_stats['source_files_names'] = source_files_names
dat_stats['total_tweets_processed'] = total_tweets_processed
dat_stats['total_tweets_removed'] = total_tweets_removed
dat_stats['num_tweets_map'] = num_tweets_map
dat_stats['unigram_count'] = unigram_count
print(dat_stats)
pickle.dump(dat_stats, open(os.path.join(dest_full_path, 'data', 'dat_stats.pkl'), 'wb'))


