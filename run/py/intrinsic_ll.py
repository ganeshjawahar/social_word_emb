# calculates the likelihood (l_pos + l_neg) on held-out dev/test samples for baselines
# measure based on https://arxiv.org/pdf/1703.08052.pdf

import sys
import os
import argparse
import pickle
from tqdm import tqdm
import glob
import codecs

import numpy as np
import torch
from torch.autograd import Variable
torch.manual_seed(123)

parser = argparse.ArgumentParser(description="evaluation script for getting log likelihood scores")
parser.add_argument("--data_dir", type=str, default="/home/ganesh/data/sosweet/full", help="directory containing raw data")
parser.add_argument("--tensor_dir", type=str, default="/home/ganesh/objects/dywoev/aug1/full", help="directory containing data tensors")
parser.add_argument("--neg_dir", type=str, default="/home/ganesh/data/sosweet/full/intrinsic/ll_neg", help="dir where negative samples are present")
parser.add_argument("--model_dir", type=str, default="/home/ganesh/objects/dywoev/baselines/word2vec/hamilton_new", help="directory containing trained dynamic word embeddings")
parser.add_argument('--bsize', type=int, default=500, help='batch size for inference')
parser.add_argument("--cuda", action='store_true', help="use gpu")
args = parser.parse_args()
print(args)

class data():
  def __init__(self, path, neg_dir, batch_size, source_files_names):
    self.src_path = path
    self.neg_dir = neg_dir

    # read data stats
    dat_stats = pickle.load(open(os.path.join(self.src_path, 'data', 'dat_stats.pkl'), 'rb'))
    self.source_files_names = source_files_names
    self.n_valid, self.n_test = [], []
    for i in range(len(dat_stats['num_tweets_map'])):
      self.n_valid.append(dat_stats['num_tweets_map'][i][3])
      self.n_test.append(dat_stats['num_tweets_map'][i][4])
    self.n_valid, self.n_test = np.array(self.n_valid, dtype=np.int32), np.array(self.n_test, dtype=np.int32)
    self.context_size = dat_stats['context_size']

    # data generator
    self.valid_batch, self.test_batch = [], []
    self.valid_neg_batch, self.test_neg_batch = [], []
    for t, files in enumerate(self.source_files_names):
      self.valid_batch.append(self.batch_generator(batch_size, 'dev', files)) # batch size is fixed from train set
      self.test_batch.append(self.batch_generator(batch_size, 'test', files)) # batch size is fixed from train set
      self.valid_neg_batch.append(self.neg_generator(batch_size, 'dev', files))
      self.test_neg_batch.append(self.neg_generator(batch_size, 'test', files))

  def neg_generator(self, batch_size, folder, files):
    neg_dir = self.neg_dir
    assert(os.path.exists(neg_dir)==True)
    f_idx = 0
    data = pickle.load(open(os.path.join(neg_dir, folder, files[f_idx]+".pkl"), 'rb'))
    cur_idx = 0
    record_size = 20
    num_records = len(data)//record_size
    while True:
      cur_batch, size = [], 0
      while size < batch_size:
        while size < batch_size and cur_idx < num_records:
          cur_batch.append(data[ cur_idx*record_size: (cur_idx+1)*record_size ])
          cur_idx = cur_idx + 1
          size = size + 1
        if size < batch_size:
          assert(len(data)==(num_records*record_size))
          f_idx=(f_idx+1)%len(files)
          data = pickle.load(open(os.path.join(neg_dir, folder, files[f_idx]+".pkl"), 'rb'))
          cur_idx = 0
          num_records = len(data)//record_size
      assert(len(cur_batch)==batch_size)
      yield cur_batch

  def batch_generator(self, batch_size, folder, files):
    f_idx = 0
    data = pickle.load(open(os.path.join(self.src_path, 'data', folder, files[f_idx]+'.pkl'), 'rb'))
    cur_idx = 0
    rand_idxs = np.random.permutation(len(data))
    while True:
      cur_batch, size = [], 0
      while size < batch_size:
        while size < batch_size and cur_idx < len(data):
          cur_batch.append(data[rand_idxs[cur_idx]])
          cur_idx = cur_idx + 1
          size = size + 1
        if size < batch_size:
          assert(cur_idx==len(data))
          f_idx=(f_idx+1)%len(files)
          data = pickle.load(open(os.path.join(self.src_path, 'data', folder, files[f_idx]+'.pkl'), 'rb'))
          cur_idx = 0
          rand_idxs = np.random.permutation(len(data))
      assert(len(cur_batch)==batch_size)
      yield cur_batch

print('loading vocab id maps...')
word2id, id2word = {}, {}
with codecs.open(os.path.join(args.data_dir, 'main', 'vocab_25K'), 'r', 'utf-8') as f:
  for word_row in f:
    word_id, word_str, count = word_row.strip().split("\t")
    word_id, count = int(word_id), int(count)
    word2id[word_str] = word_id
    id2word[word_id] = word_str

print('loading trained embeddings...')
train_embed = pickle.load(open(os.path.join(args.model_dir, 'stats.pkl'), 'rb'))
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
  mean_target_embeds[cur_time] = torch.from_numpy(mean_target_embed)
  mean_context_embeds[cur_time] = torch.from_numpy(mean_context_embed)

print('reading data...')
data = data(args.tensor_dir, args.neg_dir, args.bsize, time_period)

batch_target, batch_context = [], []
batch_neg_target = []
for t in range(len(data.n_valid)):
  bt_tensor = torch.FloatTensor(args.bsize, 1, word_dim)
  batch_target.append(bt_tensor)
  bc_tensor = torch.FloatTensor(args.bsize, 2 * context_size, word_dim)
  batch_context.append(bc_tensor)
  bnt_tensor = torch.FloatTensor(args.bsize, 20, word_dim)
  batch_neg_target.append(bnt_tensor)
  if args.cuda > 0:
    batch_target[t] = batch_target[t].cuda()
    batch_context[t] = batch_context[t].cuda()
    batch_neg_target[t] = batch_neg_target[t].cuda()

bern_val_pos = torch.ones(args.bsize, 1)
bern_val_neg = torch.zeros(args.bsize, 20)
zero_pad_vec = torch.zeros(word_dim)
if args.cuda:
  bern_val_pos = bern_val_pos.cuda()
  bern_val_neg = bern_val_neg.cuda()
  zero_pad_vec = zero_pad_vec.cuda()

def model_eval(contexts, targets, neg_targets, eval_range):
  ll_total_master = []
  for i, t in enumerate(eval_range):
    #cur_bsize = contexts[i].data.size(0)

    # pos samples
    pos_context, pos_target = contexts[i], targets[i]
    pos_context_sum = torch.unsqueeze(torch.sum(pos_context, 1), 2)
    pos_eta = torch.squeeze(torch.matmul(pos_target, pos_context_sum), 1)
    y_pos = torch.distributions.Bernoulli(logits=pos_eta)
    ll_pos = y_pos.log_prob(Variable(bern_val_pos))
    if args.cuda:
      ll_pos = ll_pos.cuda()

    # neg samples
    neg_target = neg_targets[i]
    neg_eta = torch.squeeze(torch.matmul(neg_target, pos_context_sum))
    y_neg = torch.distributions.Bernoulli(logits=neg_eta)
    ll_neg = y_neg.log_prob(Variable(bern_val_neg))
    if args.cuda:
      ll_neg = ll_neg.cuda()
    ll_neg = ll_neg.sum(1)

    ll_total = ll_pos.squeeze()+ll_neg
    ll_total_master.append(ll_total)
  return ll_total_master

def evaluate(batch_gen, num_instances, batch_size, context_size, neg_batch_gen):
  num_batches = np.amax(np.ceil((1.0*num_instances)/batch_size)).astype('int32')
  T = len(batch_gen)
  scores, num_processed = {}, [0]*T
  for t in range(T):
    scores[t] = np.array([0.0]*num_instances[t])
  for bi in tqdm(range(num_batches)):
    cur_context, cur_target, eval_range = [], [], []
    cur_neg_target = []
    for t in range(T):
      if num_processed[t] < num_instances[t]:
        eval_range.append(t)
        cur_batch = next(batch_gen[t])
        # prepare input
        for i, inp in enumerate(cur_batch):
          batch_target[t][i][0] = torch.from_numpy(target_embeds[t][id2word[inp[0]]]) if id2word[inp[0]] in target_embeds[t] else mean_target_embeds[t]
          for j in range(2 * context_size):
            if inp[1][j] >= len(word2id):
              batch_context[t][i][j] = zero_pad_vec
            elif id2word[inp[1][j]] in context_embeds[t]:
              batch_context[t][i][j] = torch.from_numpy(context_embeds[t][id2word[inp[1][j]]])
            else:
              batch_context[t][i][j] = mean_context_embeds[t]

        cur_context.append(Variable(batch_context[t], requires_grad=False))
        cur_target.append(Variable(batch_target[t], requires_grad=False))
        cur_neg_batch = next(neg_batch_gen[t])
        for i, inp in enumerate(cur_neg_batch):
          for j in range(20):
            batch_neg_target[t][i][j] = torch.from_numpy(target_embeds[t][id2word[inp[j]]]) if id2word[inp[j]] in target_embeds[t] else mean_target_embeds[t]
        cur_neg_target.append(Variable(batch_neg_target[t], requires_grad=False))
    assert(len(eval_range)!=0)
    ll_tot = model_eval(cur_context, cur_target, cur_neg_target, eval_range=eval_range)
    for i, t in enumerate(eval_range):
      assert(batch_target[t].size()[0]==ll_tot[i].size()[0])
      cur_bsize = min(num_instances[t], (bi+1) * batch_size[t]) - (bi * batch_size[t])
      scores[t][num_processed[t]:num_processed[t] + cur_bsize] = ll_tot[i].data.cpu().numpy()[0:cur_bsize]
      num_processed[t] = num_processed[t] + cur_bsize
  num_processed = np.array(num_processed)
  #assert(np.array_equal(num_processed, num_instances))
  final_mean, final_std = np.array([0.0]*T), np.array([0.0]*T)
  subscore = ''
  for t in range(T):
    score = scores[t]
    final_mean[t] = score.mean()
    final_std[t] = score.std()
    subscore+=str(t)+'='+str(final_mean[t])+' ('+str(final_std[t])+'),'
  #print(subscore)
  final_mean = np.average(final_mean, weights=num_processed)
  final_std = np.mean(final_std)/np.sqrt(num_processed.sum())
  return "%.4f (%.4f)"%(final_mean, final_std)

print('computing log-likelihood for dev')
print('%s score = %s'%('dev', evaluate(data.valid_batch, data.n_valid, np.repeat(args.bsize, len(data.valid_neg_batch)), context_size, data.valid_neg_batch)))

print('computing log-likelihood for test')
print('%s score = %s'%('test', evaluate(data.test_batch, data.n_test, np.repeat(args.bsize, len(data.test_neg_batch)), context_size, data.test_neg_batch)))














