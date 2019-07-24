'''
data handling and model classes for dynamic training
'''
import sys
import os
import pickle
import math

import torch
from torch.autograd import Variable
from tqdm import tqdm

import numpy as np
np.random.seed(123)
import random
import pandas as pd

class data():
  def __init__(self, path, n_epochs, neg_dir):
    self.src_path = path
    self.n_epochs = n_epochs
    self.neg_dir = neg_dir

    # read data stats
    dat_stats = pickle.load(open(os.path.join(self.src_path, 'data', 'dat_stats.pkl'), 'rb'))
    self.source_files_names = dat_stats['source_files_names']
    self.N, self.n_train, self.n_valid, self.n_test = 0, [], [], []
    for i in range(len(dat_stats['num_tweets_map'])):
      self.N+=dat_stats['num_tweets_map'][i][2]
      self.n_train.append(dat_stats['num_tweets_map'][i][2])
      self.n_valid.append(dat_stats['num_tweets_map'][i][3])
      self.n_test.append(dat_stats['num_tweets_map'][i][4])
    self.n_train, self.n_valid, self.n_test = np.array(self.n_train, dtype=np.int32), np.array(self.n_valid, dtype=np.int32), np.array(self.n_test, dtype=np.int32)
    self.n_train = (self.n_train/self.n_epochs).astype('int32')
    self.context_size = dat_stats['context_size']

    # load vocabulary
    self.counts = dat_stats['unigram_count']
    counts = (1.0 * self.counts / self.N) ** (3.0 / 4)
    self.unigram = counts / self.N

    # data generator
    self.train_batch, self.valid_batch, self.test_batch = [], [], []
    self.valid_neg_batch, self.test_neg_batch = [], []
    for t, files in enumerate(self.source_files_names):
      self.train_batch.append(self.batch_generator(self.n_train[t], 'train', files))
      self.valid_batch.append(self.batch_generator(self.n_train[t], 'dev', files)) # batch size is fixed from train set
      self.test_batch.append(self.batch_generator(self.n_train[t], 'test', files)) # batch size is fixed from train set
      self.valid_neg_batch.append(self.neg_generator(self.n_train[t], 'dev', files))
      self.test_neg_batch.append(self.neg_generator(self.n_train[t], 'test', files))

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

class model(torch.nn.Module):
  def __init__(self, K, sig, n_epochs, n_train, ns, unigram, context_size, init_emb, gpus):
    super(model, self).__init__()
    self.K = K
    self.n_epochs = n_epochs
    self.n_train = n_train
    self.T = len(self.n_train)
    self.ns = ns
    self.unigram = unigram
    self.L = self.unigram.shape[0]
    self.context_size = context_size
    self.init_emb = init_emb
    self.gpus = gpus

    # get pre-trained context and target embeddings
    pretrain_embeds = pickle.load(open(self.init_emb, 'rb'))
    context_init = pretrain_embeds['context']
    target_init = pretrain_embeds['target']

    # word embedding model
    for t in range(self.T):
      target_embed = torch.nn.Embedding(self.L, self.K, sparse=True)
      target_embed.weight.data = torch.from_numpy(target_init + 0.001*(np.random.randn(self.L, self.K).astype('float32')/self.K))
      self.add_module("target_embed_"+str(t), target_embed)
    #context_embed = torch.nn.Embedding(self.L+(self.context_size*2), self.K, sparse=True)
    context_embed = torch.nn.Embedding(self.L+1, self.K, sparse=True, padding_idx=self.L)
    context_embed.weight.data = torch.from_numpy(context_init)
    self.add_module("context_embed", context_embed)

    # placeholders, (neg) sampling distribution
    n_train_per_gpu = []
    for t in range(self.T):
      tot_bsize = self.n_train[t].item()
      if self.gpus > 1:
        if tot_bsize%2 == 0:
          n_train_per_gpu.append(tot_bsize//2)
        else:
          n_train_per_gpu.append((tot_bsize-1)//2)
          n_train_per_gpu.append(1+((tot_bsize-1)//2))
      else:
        n_train_per_gpu.append(tot_bsize)
    n_train_per_gpu = np.array(n_train_per_gpu)
    uniq_batch = np.unique(n_train_per_gpu)
    for t in range(len(uniq_batch)):
      cur_bsize = uniq_batch[t].item()
      bern_val_neg = torch.zeros(cur_bsize, self.ns)
      bern_val_pos = torch.ones(cur_bsize, 1)
      unigram_logits = torch.nn.Softmax(dim=0)(Variable(torch.Tensor(self.unigram))).repeat(cur_bsize,1)
      if self.gpus>0:
        bern_val_neg = bern_val_neg.cuda()
        bern_val_pos = bern_val_pos.cuda()
        unigram_logits = unigram_logits.cuda()
      self.register_buffer("bern_val_neg_"+str(cur_bsize), bern_val_neg)
      self.register_buffer("bern_val_pos_"+str(cur_bsize), bern_val_pos)
      self.register_buffer("unigram_logits_"+str(cur_bsize), unigram_logits.data)
      self.register_buffer("bern_val_eval_neg_"+str(cur_bsize), torch.zeros(cur_bsize, 20))
    mean = torch.Tensor([0.0])
    sig = torch.Tensor([sig])
    self.register_buffer("mean", mean)
    self.register_buffer("sig", sig)
    if self.gpus>0:
      mean = mean.cuda()
      sig = sig.cuda()
    self.global_prior_dist = torch.distributions.Normal(self.mean, self.sig)
    self.local_prior_dist = torch.distributions.Normal(self.mean, self.sig/100.0)

  def forward(self, contexts, targets):
    ll_pos_master = 0
    ll_neg_master = 0
    for i, t in enumerate(range(self.T)):
      cur_bsize = contexts[i].data.size(0)

      # pos samples
      pos_context, pos_target = self.__getattr__("context_embed")(contexts[i]), self.__getattr__("target_embed_"+str(t))(targets[i])
      pos_context_sum = torch.unsqueeze(torch.sum(pos_context, 1), 2)
      pos_eta = torch.squeeze(torch.matmul(pos_target, pos_context_sum), 1)
      y_pos = torch.distributions.Bernoulli(logits=pos_eta)
      ll_pos = y_pos.log_prob(Variable(self.__getattr__("bern_val_pos_"+str(cur_bsize))))
      if self.gpus>0:
        ll_pos = ll_pos.cuda()
      ll_pos_master += ll_pos.sum(0)

      # neg samples
      neg_idx = torch.multinomial(Variable(self.__getattr__("unigram_logits_"+str(cur_bsize))), self.ns)
      neg_target = self.__getattr__("target_embed_"+str(t))(neg_idx)
      neg_eta = torch.squeeze(torch.matmul(neg_target, pos_context_sum))
      y_neg = torch.distributions.Bernoulli(logits=neg_eta)
      ll_neg = y_neg.log_prob(Variable(self.__getattr__("bern_val_neg_"+str(cur_bsize))))
      if self.gpus>0:
        ll_neg = ll_neg.cuda()
      ll_neg_master += ll_neg.sum(0).sum(0)

    # prior
    global_prior_dist = torch.distributions.Normal(self.mean, self.sig)
    local_prior_dist = torch.distributions.Normal(self.mean, self.sig/100.0)
    prior_context = global_prior_dist.log_prob(self.__getattr__("context_embed").weight.data)
    prior_target_0 = global_prior_dist.log_prob(self.__getattr__("target_embed_%d"%(self.T-1)).weight.data)
    log_prior = Variable((prior_context.sum(0) + prior_target_0.sum(0)).sum(0))
    diff = local_prior_dist.log_prob(self.__getattr__("target_embed_%d"%(0)).weight.data-self.__getattr__("target_embed_%d"%(self.T-1)).weight.data)
    log_prior += Variable(diff.sum(0).sum(0))
    for t in range(1, self.T):
      diff = local_prior_dist.log_prob(self.__getattr__("target_embed_"+str(t)).weight.data-self.__getattr__("target_embed_"+str(t-1)).weight.data)
      log_prior += Variable(diff.sum(0).sum(0))

    # complete loss
    log_likelihood = ll_pos_master + ll_neg_master
    loss = - (self.n_epochs * log_likelihood + log_prior)

    return loss
  
  def eval(self, contexts, targets, neg_targets, eval_range):
    ll_total_master = []
    for i, t in enumerate(eval_range):
      cur_bsize = contexts[i].data.size(0)

      # pos samples
      pos_context, pos_target = self.__getattr__("context_embed")(contexts[i]), self.__getattr__("target_embed_"+str(t))(targets[i])
      pos_context_sum = torch.unsqueeze(torch.sum(pos_context, 1), 2)
      pos_eta = torch.squeeze(torch.matmul(pos_target, pos_context_sum), 1)
      y_pos = torch.distributions.Bernoulli(logits=pos_eta)
      ll_pos = y_pos.log_prob(Variable(self.__getattr__("bern_val_pos_"+str(cur_bsize))))
      if self.gpus>0:
        ll_pos = ll_pos.cuda()
      
      # neg samples
      neg_target = self.__getattr__("target_embed_"+str(t))(neg_targets[i])
      neg_eta = torch.squeeze(torch.matmul(neg_target, pos_context_sum))
      y_neg = torch.distributions.Bernoulli(logits=neg_eta)
      ll_neg = y_neg.log_prob(Variable(self.__getattr__("bern_val_eval_neg_"+str(cur_bsize))))
      if self.gpus>0:
        ll_neg = ll_neg.cuda()
      ll_neg = ll_neg.sum(1)

      ll_total = ll_pos.squeeze() + ll_neg
      ll_total_master.append(ll_total)
    return ll_total_master





