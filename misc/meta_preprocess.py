'''
preprocessing code for d-emb with meta info.
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

parser = argparse.ArgumentParser(description="preprocessing script for dynamic word embeddings with meta features...")
parser.add_argument("--src_dir", type=str, default="/home/ganesh/data/sosweet/full", help="source dir file")
parser.add_argument("--permut", action='store_true', help="randomly shuffle data across time")
parser.add_argument("--dest_dir", type=str, default="/home/ganesh/objects/dywoev/meta_monthly", help="dest dir file")
parser.add_argument("--context_size", type=int, default=2, help="max distance between focus and context word")
parser.add_argument("--dry_run", action='store_true', help="run for a small dataset")
parser.add_argument("--span_unit", type=str, default='yearly', help="monthly or quarterly or halfyearly or yearly")
parser.add_argument("--meta_id", type=str, default="user", help="user or tweet level meta information?")
parser.add_argument("--meta_type", type=str, default="embed", help="type of meta information. embed or category or presence?")
parser.add_argument("--meta_path", type=str, default="network/feat.txt", help="relative path for the meta feature file") 


args = parser.parse_args()
print(args)

'''
utility functions
'''

def gen_windows(record):
  numpy_obj, meta_id = record
  num_inputs = (1 + numpy_obj.shape[0] - (2 * args.context_size + 1))
  inputs = []
  for inp in range(num_inputs):
    target = numpy_obj[inp + args.context_size]
    left_context = numpy_obj[inp : inp + args.context_size]
    right_context = numpy_obj[inp+args.context_size+1: inp + (2 * args.context_size + 1)]
    inputs.append([target.item(), np.concatenate([left_context, right_context]), meta_id])
  return inputs

def padRecord(rec_np):
  items = []
  # left pad items
  items += [len(word2id) for i in range(args.context_size)]
  # actual tweet
  items += rec_np[0].tolist()
  # right pad items
  items += [len(word2id) for i in range(args.context_size)]
  return [np.array(items), rec_np[1]]

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
dest_full_path = args.dest_dir
if not os.path.exists(dest_full_path):
  os.makedirs(dest_full_path)
  os.makedirs(os.path.join(dest_full_path, 'data', 'train'))
  os.makedirs(os.path.join(dest_full_path, 'data', 'dev'))
  os.makedirs(os.path.join(dest_full_path, 'data', 'test'))
print('storing everything in '+dest_full_path)

# get all data files
source_files_abs = glob.glob(os.path.join(args.src_dir, 'main', 'train', '*'))
cand_files_names = []
for file in source_files_abs:
  cand_files_names.append(file.split("/")[-1])
cand_files_names = sorted(cand_files_names, key=lambda x: datetime.datetime.strptime(x, '%Y-%m'))
source_files_names = getFilesByUnit(cand_files_names)
if args.permut == True:
  random.shuffle(source_files_names)
print('source files (in order)...')
print(source_files_names)

print('loading vocab id maps...')
word2id, id2word, cnt = {}, {}, []
with codecs.open(os.path.join(args.src_dir, 'main', 'vocab_25K'), 'r', 'utf-8') as f:
  for word_row in f:
    word_id, word_str, count = word_row.strip().split("\t")
    word_id, count = int(word_id), int(count)
    word2id[word_str] = word_id
    id2word[word_id] = word_str
    cnt.append(count)
cnt = np.array(cnt)
unigram_count = cnt
cnt = 1.0*cnt/cnt.sum()

print('reading meta features...')
tweetid2metaid, tweetid2userid  = {}, {}
with open(os.path.join(args.src_dir, 'main', 'geo_tweets_full.tsv'), 'r') as f:
  for line in f:
    tweet_id, user_id = line.strip().split("\t")
    tweetid2userid[tweet_id] = user_id
if args.meta_type == 'embed':
  embeds, embed_dim = {}, None
  with open(os.path.join(args.src_dir, 'features', args.meta_id, args.meta_path), 'r') as f:
    for line in f:
      meta_id, user_embed = line.strip().split("\t")
      if not embed_dim:
        embed_dim = len(user_embed.split())
        print('meta embedding size = %d'%(embed_dim))
      embed = np.empty([1, embed_dim], dtype=np.float32)
      embed[:] = user_embed.split()
      embeds[meta_id] = embed
  save_obj = {}
  save_obj['dim'] = embed_dim
  save_obj['size'] = len(embeds)
  save_obj['embeds'] = embeds
  pickle.dump(save_obj, open(os.path.join(dest_full_path, 'data', 'meta_embed.pkl'), 'wb'))
  for tweet_id in tweetid2userid:
    user_id = tweetid2userid[tweet_id]
    if user_id in embeds:
      tweetid2metaid[tweet_id] = user_id
if args.meta_type == 'category':
  userid2metacat, catname2id, catid2name = {}, {}, {}
  with open(os.path.join(args.src_dir, 'features', args.meta_id, args.meta_path), 'r') as f:
    for line in f:
      content = line.strip().split("\t")
      meta_id, meta_category = content[0], content[-1]
      if args.meta_id == 'tweet':
        tweet_id = meta_id
        if tweet_id in tweetid2userid:
          tweetid2metaid[tweet_id] = meta_category
      if args.meta_id == 'user':
        user_id = meta_id
        userid2metacat[user_id] = meta_category
      if meta_category not in catname2id:
        catname2id[meta_category] = 1 + len(catid2name)
        catid2name[catname2id[meta_category]] = meta_category
  if len(userid2metacat) > 0:
    for tweet_id in tweetid2userid:
      user_id = tweetid2userid[tweet_id]
      if user_id in userid2metacat:
        tweetid2metaid[tweet_id] = userid2metacat[user_id]
  pickle.dump({'catname2id': catname2id, 'catid2name': catid2name}, open(os.path.join(dest_full_path, 'data', 'meta_category.pkl'), 'wb'))
if args.meta_type == 'presence':
  userid2seq, userseq2id = {}, {}
  with open(os.path.join(args.src_dir, 'features', args.meta_id, args.meta_path), 'r') as f:
    for line in f:
      user_id = line.strip()
      userid2seq[user_id] = 1 + len(userseq2id)
      userseq2id[userid2seq[user_id]] = user_id
  for tweet_id in tweetid2userid:
    user_id = tweetid2userid[tweet_id]
    if user_id in userid2seq:
      tweetid2metaid[tweet_id] = user_id
  pickle.dump({'userid2seq': userid2seq, 'userseq2id': userseq2id}, open(os.path.join(dest_full_path, 'data', 'meta_presence.pkl'), 'wb'))

print('obtained meta features for %d tweets (%.2f percent)'%(len(tweetid2metaid), (1.0*len(tweetid2metaid))/len(tweetid2userid)) )

print('creating pickle objects...')
pbar = tqdm(total=len(cand_files_names))
total_tweets_processed, total_tweets_removed = 0, 0
num_tweets_map = []
for source_file_name_arr in source_files_names:
  cur_tweets_processed, cur_tweets_removed, num_split = 0, 0, [0]*3
  for source_file_name in source_file_name_arr:
    pbar.update(1)
    cand_folders = ['train', 'dev', 'test']
    cand_size = [300, 30, 30]
    num_words_master = []
    ci = 0
    for cand_folder in cand_folders:
      source_file_abs = os.path.join(args.src_dir, 'main', cand_folder, source_file_name)
      input_plain = []
      cand_tweets_processed = 0

      with codecs.open(source_file_abs, 'r', 'utf-8', errors='ignore') as file:
        for line in file:
          if args.dry_run == True and cand_tweets_processed==cand_size[ci]:
            break

          items = line.strip().split("\t")
          cur_words = items[1].split()
          tweet_id = items[0]
          cur_meta_id = tweetid2metaid[tweet_id] if tweet_id in tweetid2metaid else None

          total_tweets_processed+=1

          cur_vocab_words = []
          for word in cur_words:
            if word in word2id:
              cur_vocab_words.append(word)
            
          cur_tweet = []
          for word in cur_vocab_words:
            cur_tweet.append(word2id[word])

          if len(cur_tweet)==0:
            total_tweets_removed+=1
            cur_tweets_removed+=1
            continue

          cur_tweet = np.array(cur_tweet)
          
          if ci == 0:
            prob = np.random.uniform(0,1,cur_tweet.shape)
            p = 1 - np.sqrt((10.0**(-5))/cnt[cur_tweet])
            cur_tweet = cur_tweet[prob > p] 
          
          input_plain.append([cur_tweet, cur_meta_id])

          cand_tweets_processed+=1
      cur_tweets_processed+=cand_tweets_processed
      records, num_words = process_data(input_plain, pad=True)
      pickle.dump(records, open(os.path.join(dest_full_path, 'data', cand_folder, source_file_name + '.pkl' ), 'wb'))
      num_split[ci]+=num_words
      ci = ci + 1
  num_tweets_map.append([cur_tweets_processed, cur_tweets_removed] + num_split)
pbar.close()

# write dat stats
dat_stats = {}
dat_stats['permut'] = args.permut
dat_stats['context_size'] = args.context_size
dat_stats['source_files_names'] = source_files_names
dat_stats['total_tweets_processed'] = total_tweets_processed
dat_stats['total_tweets_removed'] = total_tweets_removed
dat_stats['num_tweets_map'] = num_tweets_map
dat_stats['unigram_count'] = unigram_count
print(dat_stats)
pickle.dump(dat_stats, open(os.path.join(dest_full_path, 'data', 'dat_stats.pkl'), 'wb'))


