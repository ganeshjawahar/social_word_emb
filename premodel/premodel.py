'''
data handling and model classes for non-dynamic pre-training
'''
import sys
import os
import pickle
import math

import torch
from torch.autograd import Variable

import numpy as np
np.random.seed(123)
import pandas as pd

class data():
  def __init__(self, path, n_epochs, neg_dir):
    self.src_path = path
    self.n_epochs = n_epochs
    self.neg_dir = neg_dir

    # read data stats
    dat_stats = pickle.load(open(os.path.join(self.src_path, 'data', 'dat_stats.pkl'), 'rb'))
    self.source_files_names = dat_stats['source_files_names']
    self.N, self.n_train, self.n_valid, self.n_test = 0, 0, np.array([0]*len(self.source_files_names)), np.array([0]*len(self.source_files_names))
    for i in range(len(dat_stats['num_tweets_map'])):
      self.N+=dat_stats['num_tweets_map'][i][2]
      self.n_valid[i] = dat_stats['num_tweets_map'][i][3]
      self.n_test[i] = dat_stats['num_tweets_map'][i][4]
    self.n_train = int(self.N/self.n_epochs)
    self.context_size = dat_stats['context_size']

    # load vocabulary
    self.counts = dat_stats['unigram_count']
    counts = (1.0 * self.counts / self.N) ** (3.0 / 4)
    self.unigram = counts / self.N

    # data generator
    self.source_files_names_unpacked = []
    for time_data in self.source_files_names:
      self.source_files_names_unpacked+=time_data

    self.train_batch = self.batch_generator(self.n_train, 'train')
    self.valid_batch = self.batch_generator(self.n_train, 'dev') # batch size is fixed from train set
    self.test_batch = self.batch_generator(self.n_train, 'test') # batch size is fixed from train set
    self.valid_neg_batch = self.neg_generator(self.n_train, self.n_valid, 'dev')
    self.test_neg_batch = self.neg_generator(self.n_train, self.n_test, 'test')

  def neg_generator(self, batch_size, num_instances, folder):
    neg_dir = self.neg_dir
    assert(os.path.exists(neg_dir)==True)
    f_idx = 0
    data = pickle.load(open(os.path.join(neg_dir, folder, self.source_files_names_unpacked[f_idx]+".pkl"), 'rb'))
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
          f_idx=(f_idx+1)%len(self.source_files_names_unpacked)
          data = pickle.load(open(os.path.join(neg_dir, folder, self.source_files_names_unpacked[f_idx]+'.pkl'), 'rb'))
          cur_idx = 0
          num_records = len(data)//record_size
      assert(len(cur_batch)==batch_size)
      yield cur_batch

  def batch_generator(self, batch_size, folder):
    f_idx = 0
    data = pickle.load(open(os.path.join(self.src_path, 'data', folder, self.source_files_names_unpacked[f_idx]+'.pkl'), 'rb'))
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
          f_idx=(f_idx+1)%len(self.source_files_names_unpacked)
          data = pickle.load(open(os.path.join(self.src_path, 'data', folder, self.source_files_names_unpacked[f_idx]+'.pkl'), 'rb'))
          cur_idx = 0
          rand_idxs = np.random.permutation(len(data))
      assert(len(cur_batch)==batch_size)
      yield cur_batch

class model(torch.nn.Module):
  def __init__(self, K, sig, n_epochs, n_train, ns, unigram, context_size, gpus):
    super(model, self).__init__()
    self.K = K
    self.n_epochs = n_epochs
    self.n_train = n_train
    self.ns = ns
    self.unigram = unigram
    self.L = self.unigram.shape[0]
    self.context_size = context_size
    self.gpus = gpus

    # word embeddings
    self.target_embed = torch.nn.Embedding(self.L, self.K, sparse=True)
    #self.context_embed = torch.nn.Embedding(self.L+(self.context_size*2), self.K, sparse=True)
    self.context_embed = torch.nn.Embedding(self.L+1, self.K, sparse=True, padding_idx=self.L)
    
    # random init
    self.target_embed.weight.data = torch.randn(self.L, self.K)/self.K
    #self.context_embed.weight.data = torch.randn(self.L+(self.context_size*2), self.K)/self.K
    self.context_embed.weight.data = torch.randn(self.L+1, self.K)/self.K

    # placeholders and (neg) sampling distribution
    n_train_per_gpu = [self.n_train]
    if self.gpus>1:
      n_train_per_gpu += [self.n_train//2] if self.n_train%2==0 else [ (self.n_train-1)//2, 1+(self.n_train-1)//2 ]

    for bsize in n_train_per_gpu:
      bern_val_neg = torch.zeros(bsize, self.ns)
      bern_val_pos = torch.ones(bsize, 1)
      unigram_logits = torch.nn.Softmax(dim=0)(Variable(torch.Tensor(self.unigram))).repeat(bsize,1)
      self.register_buffer("bern_val_neg_"+str(bsize), bern_val_neg)
      self.register_buffer("bern_val_pos_"+str(bsize), bern_val_pos)
      self.register_buffer("unigram_logits_"+str(bsize), unigram_logits.data)
      self.register_buffer("bern_val_eval_neg_"+str(bsize), torch.zeros(bsize, 20))

    self.prior_dist = torch.distributions.Normal(torch.Tensor([0.0]).cuda(), torch.Tensor([sig]).cuda()) if self.gpus>0 else torch.distributions.Normal(torch.Tensor([0.0]), torch.Tensor([sig]))

  def forward(self, contexts, targets):
    cur_bsize = contexts.data.size(0)

    # pos samples
    pos_context, pos_target = self.context_embed(contexts), self.target_embed(targets)
    pos_context_sum = torch.unsqueeze(torch.sum(pos_context, 1), 2)
    pos_eta = torch.squeeze(torch.matmul(pos_target, pos_context_sum), 1)
    y_pos = torch.distributions.Bernoulli(logits=pos_eta)
    ll_pos = y_pos.log_prob(Variable(self.__getattr__("bern_val_pos_"+str(cur_bsize))))
    if self.gpus>0:
      ll_pos = ll_pos.cuda()
    ll_pos = ll_pos.sum(0)

    # neg samples 
    neg_idx = torch.multinomial(Variable(self.__getattr__("unigram_logits_"+str(cur_bsize))), self.ns)
    neg_target = self.target_embed(neg_idx)
    neg_eta = torch.squeeze(torch.matmul(neg_target, pos_context_sum))
    y_neg = torch.distributions.Bernoulli(logits=neg_eta)
    ll_neg = y_neg.log_prob(Variable(self.__getattr__("bern_val_neg_"+str(cur_bsize))))
    if self.gpus>0:
      ll_neg = ll_neg.cuda()
    ll_neg = ll_neg.sum(0).sum(0)

    # prior
    prior_context = self.prior_dist.log_prob(self.context_embed.weight.data)
    prior_target = self.prior_dist.log_prob(self.target_embed.weight.data)
    log_prior = (prior_context.sum(0) + prior_target.sum(0))
    log_prior = Variable(log_prior.sum(0))

    # complete loss
    log_likelihood = ll_pos + ll_neg
    loss = - (self.n_epochs * log_likelihood + log_prior)

    return loss

  def eval(self, contexts, targets, neg_targets):
    cur_bsize = contexts.data.size(0)

    # pos samples
    pos_context, pos_target = self.context_embed(contexts), self.target_embed(targets)
    pos_context_sum = torch.unsqueeze(torch.sum(pos_context, 1), 2)
    pos_eta = torch.squeeze(torch.matmul(pos_target, pos_context_sum), 1)
    y_pos = torch.distributions.Bernoulli(logits=pos_eta)
    ll_pos = y_pos.log_prob(Variable(self.__getattr__("bern_val_pos_"+str(cur_bsize))))
    if self.gpus>0:
      ll_pos = ll_pos.cuda()

    # neg samples 
    neg_targets = self.target_embed(neg_targets)
    neg_eta = torch.squeeze(torch.matmul(neg_targets, pos_context_sum))
    y_neg = torch.distributions.Bernoulli(logits=neg_eta)
    ll_neg = y_neg.log_prob(Variable(self.__getattr__("bern_val_eval_neg_"+str(cur_bsize))))
    if self.gpus>0:
      ll_neg = ll_neg.cuda()
    ll_neg = ll_neg.sum(1)
    
    ll_total = ll_pos.squeeze() + ll_neg
    return ll_total






