'''
data handling and model classes for dynamic training
'''
import sys
import os
import pickle
import math
import time

import torch
from torch.autograd import Variable
from tqdm import tqdm
from torch.nn import Parameter, ParameterList

import numpy as np
np.random.seed(123)
import random
import pandas as pd

class data():
  def __init__(self, args):
    self.src_path = args.source_dir
    self.n_epochs = args.n_epochs
    self.neg_dir = args.neg_dir

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

    if not args.use_meta:
      print('Disclaimer: Will read the meta features but it wont be used for training and testing')
    '''
    self.reg_meta = False
    # load meta embedding (if available)
    meta_embed_pkl_f = os.path.join(self.src_path, 'data', 'meta_embed.pkl')
    if os.path.exists(meta_embed_pkl_f):
      meta_embed_content = pickle.load(open(meta_embed_pkl_f, 'rb'))
      self.meta_dim, self.meta_size, self.meta_embeds = meta_embed_content['dim'], meta_embed_content['size'], meta_embed_content['embeds']
      self.id2meta, self.meta2id, self.meta_npy = {}, {}, [np.zeros(self.meta_dim)]
      for meta_item in self.meta_embeds:
        self.meta2id[meta_item] = 1 + len(self.id2meta)
        self.id2meta[self.meta2id[meta_item]] = meta_item
        self.meta_npy.append(np.squeeze(self.meta_embeds[meta_item]))
      self.meta_npy = np.array(self.meta_npy)

    # load meta category (if available)
    meta_cat_pkl_f = os.path.join(self.src_path, 'data', 'meta_category.pkl')
    if os.path.exists(meta_cat_pkl_f):
      meta_cat_content = pickle.load(open(meta_cat_pkl_f, 'rb'))
      self.id2meta, self.meta2id = meta_cat_content['catid2name'], meta_cat_content['catname2id']
      self.reg_meta = True
    
    # load meta presence (if available)
    meta_pres_pkl_f = os.path.join(self.src_path, 'data', 'meta_presence.pkl')
    if os.path.exists(meta_pres_pkl_f):
      meta_pres_content = pickle.load(open(meta_pres_pkl_f, 'rb'))
      self.id2meta, self.meta2id = meta_pres_content['userseq2id'], meta_pres_content['userid2seq']
      self.reg_meta = True
    '''

    #if 'meta_ids' in dat_stats:
    self.multi_meta = True
    self.meta_master = []
    self.reg_meta = []
    meta_ids = args.meta_ids.split(",")
    meta_types = args.meta_types.split(",")
    meta_paths = args.meta_paths.split(",")
    metastr2seq, seq2metastr = dat_stats['metastr2seq'], dat_stats['seq2metastr']
    self.metaseq = []
    for meta_id, meta_type, meta_path in zip(meta_ids, meta_types, meta_paths):
      cur_meta_str = 'meta_%s_%s_%s.pkl'%(meta_id, meta_type, meta_path.replace("/", "_"))
      meta_pkl_f = os.path.join(self.src_path, 'data', cur_meta_str)
      assert(os.path.exists(meta_pkl_f))
      print('loading %s'%meta_pkl_f)
      if meta_type == 'embed':
        meta_embed_content = pickle.load(open(meta_pkl_f, 'rb'))
        meta_dim, meta_size, meta_embeds = meta_embed_content['dim'], meta_embed_content['size'], meta_embed_content['embeds']
        id2meta, meta2id, meta_npy = {}, {}, [np.zeros(meta_dim)]
        for meta_item in meta_embeds:
          meta2id[meta_item] = 1 + len(id2meta)
          id2meta[meta2id[meta_item]] = meta_item
          meta_npy.append(np.squeeze(meta_embeds[meta_item]))
        meta_npy = np.array(meta_npy)
        self.meta_master.append({'meta_npy': meta_npy, 'id2meta': id2meta, 'meta2id': meta2id, 'meta_dim':meta_dim, 'meta_size':meta_size})
        self.reg_meta.append(False)
      elif meta_type == 'category':
        meta_cat_content = pickle.load(open(meta_pkl_f, 'rb'))
        self.meta_master.append({'id2meta': meta_cat_content['catid2name'], 'meta2id': meta_cat_content['catname2id'], 'meta_dim': args.K})
        self.reg_meta.append(True)
      elif meta_type == 'presence':
        meta_pres_content = pickle.load(open(meta_pkl_f, 'rb'))
        self.meta_master.append({'id2meta': meta_pres_content['userseq2id'], 'meta2id': meta_pres_content['userid2seq'], 'meta_dim': args.K})
        self.reg_meta.append(True)
      self.metaseq.append(metastr2seq[meta_path.replace("/", "_")])

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
  def __init__(self, args, data):
    super(model, self).__init__()
    self.K = args.K
    self.n_epochs = args.n_epochs
    self.n_train = data.n_train
    self.T = len(self.n_train)
    self.ns = args.ns
    self.unigram = data.unigram
    self.L = self.unigram.shape[0]
    self.context_size = data.context_size
    self.init_emb = args.init_emb
    self.gpus = args.gpus
    self.sparse_modules, self.dense_modules = [], []
    self.use_meta = args.use_meta
    self.use_hierarchical = args.use_hierarchical
    self.use_amortized = args.use_amortized
    self.sharing_time = args.sharing_time
    self.meta_compo = args.meta_compo
    self.attn_type = args.attn_type
    self.multi_meta = data.multi_meta
    self.reg_meta = data.reg_meta

    # get pre-trained context and target embeddings
    context_init, target_init = None, None
    if os.path.exists(self.init_emb):
      print('loading pre-trained embeddings from %s'%self.init_emb)
      pretrain_embeds = pickle.load(open(self.init_emb, 'rb'))
      context_init = pretrain_embeds['context']
      target_init = pretrain_embeds['target']

    if args.use_meta:
      # meta embeddings
      if not data.multi_meta:
        self.meta_dim = None
        if hasattr(data, 'meta_dim'):
          meta_embed = torch.nn.Embedding(data.meta_size+1, data.meta_dim, sparse=True, padding_idx=0)
          meta_embed.weight.data = torch.from_numpy(data.meta_npy).float()
          meta_embed.weight.requires_grad = False
          self.add_module("meta_embed", meta_embed)
          self.meta_dim = data.meta_dim
          self.meta_size = data.meta_size+1
        else:
          # meta category/presence
          meta_embed = torch.nn.Embedding(len(data.id2meta)+1, self.K, sparse=True, padding_idx=0)
          self.add_module("meta_embed", meta_embed)
          self.sparse_modules.append("meta_embed")
          self.meta_dim = self.K
          self.meta_size = len(data.id2meta)+1

        # phi - modulation network
        mod_net = torch.nn.Sequential(torch.nn.Linear(self.meta_dim, self.K, False), torch.nn.Tanh(),torch.nn.Linear(self.K, self.K, False))
        self.add_module("mod_net", mod_net)
        self.dense_modules.append("mod_net")

        # meta composition parameters
        meta_out = torch.nn.Linear(self.K ,self.K, False)
        self.add_module("meta_out", meta_out)
        self.dense_modules.append("meta_out")
        if self.meta_compo == 'bilinear':
          meta_bilinear = torch.nn.Bilinear(self.K, self.K, self.K, False)
          self.add_module("meta_bilinear", meta_bilinear)
          self.dense_modules.append("meta_bilinear")
      else:
        self.num_meta = len(data.meta_master)
        for mi, meta_info in enumerate(data.meta_master):
          if 'meta_size' in meta_info:
            meta_embed = torch.nn.Embedding(meta_info['meta_size']+1, meta_info['meta_dim'], sparse=True, padding_idx=0)
            meta_embed.weight.data = torch.from_numpy(meta_info['meta_npy']).float()
            meta_embed.weight.requires_grad = False
            self.add_module("meta_embed_%d"%mi, meta_embed)
          else:
            # meta category/presence
            meta_embed = torch.nn.Embedding(len(meta_info['id2meta'])+1, meta_info['meta_dim'], sparse=True)#, padding_idx=0)
            self.add_module("meta_embed_%d"%mi, meta_embed)
            #print(len(meta_info['id2meta']), meta_info['meta_dim'])
            self.sparse_modules.append("meta_embed_%d"%mi)
          proj_net = torch.nn.Linear(meta_info['meta_dim'] if 'meta_dim' in meta_info else self.K, self.K, True)
          self.add_module("proj_net_%d"%mi, proj_net)
          self.dense_modules.append("proj_net_%d"%mi)
        if self.attn_type != 'naive':
          if self.attn_type == 'self':
            attention_weight = ParameterList([Parameter(torch.FloatTensor([0.0]*self.K))])
          else:
            attention_weight = ParameterList([Parameter(torch.FloatTensor([0.0]*self.K)) for mi in range(len(data.meta_master))])
          self.add_module("attention_weight", attention_weight)
          self.dense_modules.append('attention_weight')    
          attention_bias = ParameterList([Parameter(torch.FloatTensor([1.0]))])
          self.add_module("attention_bias", attention_bias)
          self.dense_modules.append('attention_bias')

    # word embeddings
    for t in range(self.T):
      target_embed = torch.nn.Embedding(self.L, self.K, sparse=True)
      if os.path.exists(self.init_emb):
        target_embed.weight.data = torch.from_numpy(target_init + 0.001*(np.random.randn(self.L, self.K).astype('float32')/self.K))
      self.add_module("target_embed_"+str(t), target_embed)
      self.sparse_modules.append("target_embed_"+str(t))
      #print(self.L, self.K)
    context_embed = torch.nn.Embedding(self.L+1, self.K, sparse=True, padding_idx=self.L)
    if os.path.exists(self.init_emb):
      context_embed.weight.data = torch.from_numpy(context_init)
    self.add_module("context_embed", context_embed)
    self.sparse_modules.append("context_embed")
    #print(self.L+1, self.K)
    '''
    if args.use_hierarchical:
      target_embed = torch.nn.Embedding(self.L, self.K, sparse=True)
      if os.path.exists(self.init_emb):
        target_embed.weight.data = torch.from_numpy(target_init + 0.001*(np.random.randn(self.L, self.K).astype('float32')/self.K))
      self.add_module("target_embed_sharing", target_embed)
      self.sparse_modules.append("target_embed_sharing")
    '''

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
      # meta_neg_logits = torch.Tensor([0.0]+([1.0/(self.meta_size-1)]*(self.meta_size-1))).repeat(cur_bsize,1)
      # self.register_buffer("meta_neg_logits_"+str(cur_bsize), meta_neg_logits)
    mean = torch.Tensor([0.0])
    sig = torch.Tensor([args.sig])
    self.register_buffer("mean", mean)
    self.register_buffer("sig", sig)
    if self.gpus>0:
      mean = mean.cuda()
      sig = sig.cuda()
    print('sparse modules =>')
    print(self.sparse_modules)
    print('dense modules =>')
    print(self.dense_modules)
    
  def forward(self, contexts, targets, metas):
    ll_pos_master = 0
    ll_neg_master = 0
    for i, t in enumerate(range(self.T)):
      cur_bsize = contexts[i].data.size(0)

      # pos samples
      pos_context, pos_target = self.__getattr__("context_embed")(contexts[i]), self.__getattr__("target_embed_"+str(t))(targets[i])
      pos_context_sum = torch.unsqueeze(torch.sum(pos_context, 1), 2)
      if self.use_meta:
        if not self.multi_meta:
          meta_feat = self.__getattr__("meta_embed")(metas[i])
          meta_feat = self.__getattr__("mod_net")(meta_feat)
          if self.meta_compo == 'add':
            pos_target = pos_target + meta_feat
          elif self.meta_compo == 'maxpool':
            pos_target = torch.max(pos_target, meta_feat)
          elif self.meta_compo == 'gateword':
            pos_target = torch.sigmoid(pos_target)*meta_feat
          elif self.meta_compo == 'gatemeta':
            pos_target = torch.sigmoid(meta_feat)*pos_target
          elif self.meta_compo == 'bilinear':
            pos_target = self.__getattr__("meta_bilinear")(pos_target, meta_feat)
          elif self.meta_compo == 'elemult':
            pos_target = pos_target * meta_feat
          pos_target = self.__getattr__("meta_out")(pos_target)
        else:
          if self.attn_type != 'naive':
            attn_alphas, meta_embeds = None, None
            attention_weight = self.__getattr__("attention_weight")
            attention_bias = self.__getattr__("attention_bias")[0]
            for mi in range(self.num_meta):
              meta_embed = self.__getattr__("meta_embed_%d"%mi)(metas[i][mi])
              meta_embed = self.__getattr__("proj_net_%d"%mi)(meta_embed)
              if self.attn_type == 'self':
                attn_alpha = torch.matmul(meta_embed, attention_weight[0].unsqueeze(1)) + attention_bias
              else:
                attn_alpha = torch.matmul(pos_target, attention_weight[mi].unsqueeze(1)) + attention_bias
              if mi == 0:
                meta_embeds = meta_embed
              else:
                meta_embeds = torch.cat([meta_embeds, meta_embed], dim=1)
              if mi == 0:
                attn_alphas = attn_alpha
              else:
                attn_alphas = torch.cat([attn_alphas, attn_alpha], dim=1)
            attn_alphas = torch.nn.Softmax(dim=1)(attn_alphas)
            meta_feat = torch.matmul(meta_embeds.reshape(cur_bsize, self.K, -1), attn_alphas).squeeze().unsqueeze(1)
          else:
            meta_feat = None
            for mi in range(self.num_meta):
              meta_embed = self.__getattr__("meta_embed_%d"%mi)(metas[i][mi])
              meta_embed = self.__getattr__("proj_net_%d"%mi)(meta_embed)
              if meta_feat is None:
                meta_feat = meta_embed
              else:
                meta_feat += meta_embed
          pos_target = pos_target + meta_feat
        #pos_target = pos_target + meta_feat
        #pos_target = torch.max(pos_target, meta_feat)
      pos_eta = torch.squeeze(torch.matmul(pos_target, pos_context_sum), 1)
      y_pos = torch.distributions.Bernoulli(logits=pos_eta)
      ll_pos = y_pos.log_prob(Variable(self.__getattr__("bern_val_pos_"+str(cur_bsize))))
      if self.gpus>0:
        ll_pos = ll_pos.cuda()
      ll_pos_master += ll_pos.sum(0)

      # neg samples
      start_time = time.time()
      neg_idx = torch.multinomial(Variable(self.__getattr__("unigram_logits_"+str(cur_bsize))), self.ns)
      #print("--- %s neg sampling seconds ---" % (time.time() - start_time))
      start_time = time.time()
      neg_target = self.__getattr__("target_embed_"+str(t))(neg_idx)   
      #print("--- %s neg sampled target forward seconds ---" % (time.time() - start_time))
      start_time = time.time()
      #if self.use_meta:
      #  meta_idx = torch.multinomial(Variable(self.__getattr__("meta_neg_logits_"+str(cur_bsize))), self.ns)
      #  meta_feat = self.__getattr__("meta_embed")(meta_idx)
      #  meta_feat = self.__getattr__("mod_net")(meta_feat)
      #  neg_target = neg_target+meta_feat
      if self.use_meta:
        if cur_bsize == 1:
          meta_feat = meta_feat.squeeze().unsqueeze(0).unsqueeze(0)
        neg_target = neg_target+meta_feat.expand(cur_bsize, self.ns, self.K)
      neg_eta = torch.squeeze(torch.matmul(neg_target, pos_context_sum))
      #print("--- %s neg matrix mult. seconds ---" % (time.time() - start_time))
      start_time = time.time()
      y_neg = torch.distributions.Bernoulli(logits=neg_eta)
      ll_neg = y_neg.log_prob(Variable(self.__getattr__("bern_val_neg_"+str(cur_bsize))))
      #print("--- %s neg prob dist seconds ---" % (time.time() - start_time))
      start_time = time.time()
      if self.gpus>0:
        ll_neg = ll_neg.cuda()
      #print("--- %s neg cudaing seconds ---" % (time.time() - start_time))
      start_time = time.time()
      ll_neg_master += ll_neg.sum()
      #print("--- %s neg sum probs seconds ---" % (time.time() - start_time))

    # global prior
    global_prior_dist = torch.distributions.Normal(self.mean, self.sig)
    local_prior_dist = torch.distributions.Normal(self.mean, self.sig/100.0)
    prior_context = global_prior_dist.log_prob(self.__getattr__("context_embed").weight.data)
    prior_target_0 = global_prior_dist.log_prob(self.__getattr__("target_embed_%d"%(self.T-1)).weight.data)
    global_prior_vals = (prior_context.sum() + prior_target_0.sum())
    if self.use_meta:
      if not self.multi_meta:
        global_prior_vals += global_prior_dist.log_prob(self.__getattr__("mod_net")[0].weight.data).sum() + global_prior_dist.log_prob(self.__getattr__("mod_net")[2].weight.data).sum() + global_prior_dist.log_prob(self.__getattr__("meta_out").weight.data).sum()
        if self.meta_compo == 'bilinear':
          global_prior_vals += global_prior_dist.log_prob(self.__getattr__("meta_bilinear").weight.data).sum()
        if self.reg_meta:
          global_prior_vals += global_prior_dist.log_prob(self.__getattr__("meta_embed").weight.data).sum()
      else:
        if not self.use_hierarchical:
          for wi in range(len(self.__getattr__("attention_weight"))):
            global_prior_vals += global_prior_dist.log_prob(self.__getattr__("attention_weight")[wi].data).sum()
          global_prior_vals += global_prior_dist.log_prob(self.__getattr__("attention_bias")[0].data).sum()
          for mi in range(self.num_meta):
            if self.reg_meta[mi]:
              global_prior_vals += global_prior_dist.log_prob(self.__getattr__("meta_embed_%d"%mi).weight.data).sum()
            global_prior_vals += global_prior_dist.log_prob(self.__getattr__("proj_net_%d"%mi).weight.data).sum()
    #if self.use_hierarchical:
    #  global_prior_vals += global_prior_dist.log_prob(self.__getattr__("target_embed_sharing").weight.data).sum(0)
    global_prior = Variable(global_prior_vals.sum(0))

    # local prior
    local_prior = None
    for t in range(self.T):
      cur_t = t
      if self.sharing_time == 'dynamic':
        prev_t = (t-1) if t!=0 else self.T-1
        diff = local_prior_dist.log_prob(self.__getattr__("target_embed_%d"%(cur_t)).weight.data-self.__getattr__("target_embed_%d"%(prev_t)).weight.data)
        if not local_prior:
          local_prior = Variable(diff.sum())
        else:
          local_prior += Variable(diff.sum())
      '''
      if self.use_hierarchical:
        diff = local_prior_dist.log_prob(self.__getattr__("target_embed_%d"%(cur_t)).weight.data-self.__getattr__("target_embed_sharing").weight.data)
        if not local_prior:
          local_prior = Variable(diff.sum())
        else:
          local_prior += Variable(diff.sum())
      '''

    # local + global
    log_prior = global_prior
    if self.sharing_time == 'dynamic':# or self.use_hierarchical:
      log_prior += local_prior

    # complete loss
    log_likelihood = ll_pos_master + ll_neg_master
    loss = - (self.n_epochs * log_likelihood + log_prior)

    return loss
  
  def eval(self, contexts, targets, neg_targets, metas, eval_range):
    ll_total_master = []
    for i, t in enumerate(eval_range):
      cur_bsize = contexts[i].data.size(0)

      # pos samples
      pos_context, pos_target = self.__getattr__("context_embed")(contexts[i]), self.__getattr__("target_embed_"+str(t))(targets[i])
      pos_context_sum = torch.unsqueeze(torch.sum(pos_context, 1), 2)
      if self.use_meta:
        if not self.multi_meta:
          meta_feat = self.__getattr__("meta_embed")(metas[i])
          meta_feat = self.__getattr__("mod_net")(meta_feat)

          if self.meta_compo == 'add':
            pos_target = pos_target + meta_feat
          elif self.meta_compo == 'maxpool':
            pos_target = torch.max(pos_target, meta_feat)
          elif self.meta_compo == 'gateword':
            pos_target = torch.sigmoid(pos_target)*meta_feat
          elif self.meta_compo == 'gatemeta':
            pos_target = torch.sigmoid(meta_feat)*pos_target
          elif self.meta_compo == 'bilinear':
            pos_target = self.__getattr__("meta_bilinear")(pos_target, meta_feat)
          elif self.meta_compo == 'elemult':
            pos_target = pos_target * meta_feat
          pos_target = self.__getattr__("meta_out")(pos_target)
        else:
          if self.attn_type != 'naive':
            attn_alphas, meta_embeds = None, None
            attention_weight = self.__getattr__("attention_weight")
            attention_bias = self.__getattr__("attention_bias")[0]
            for mi in range(self.num_meta):
              meta_embed = self.__getattr__("meta_embed_%d"%mi)(metas[i][mi])
              meta_embed = self.__getattr__("proj_net_%d"%mi)(meta_embed)
              if self.attn_type == 'self':
                attn_alpha = torch.matmul(meta_embed, attention_weight[0].unsqueeze(1)) + attention_bias
              else:
                attn_alpha = torch.matmul(pos_target, attention_weight[mi].unsqueeze(1)) + attention_bias
              if mi == 0:
                meta_embeds = meta_embed
              else:
                meta_embeds = torch.cat([meta_embeds, meta_embed], dim=1)
              attn_alpha[0] = mi 
              if mi == 0:
                attn_alphas = attn_alpha
              else:
                attn_alphas = torch.cat([attn_alphas, attn_alpha], dim=1)
            attn_alphas = torch.nn.Softmax(dim=1)(attn_alphas)
            meta_feat = torch.matmul(meta_embeds.reshape(cur_bsize, self.K, -1), attn_alphas).squeeze().unsqueeze(1)
          else:
            meta_feat = None
            for mi in range(self.num_meta):
              meta_embed = self.__getattr__("meta_embed_%d"%mi)(metas[i][mi])
              meta_embed = self.__getattr__("proj_net_%d"%mi)(meta_embed)
              if meta_feat is None:
                meta_feat = meta_embed
              else:
                meta_feat += meta_embed
          pos_target = pos_target + meta_feat

        #pos_target = pos_target + meta_feat
        #pos_target = torch.max(pos_target, meta_feat)
      pos_eta = torch.squeeze(torch.matmul(pos_target, pos_context_sum), 1)
      y_pos = torch.distributions.Bernoulli(logits=pos_eta)
      ll_pos = y_pos.log_prob(Variable(self.__getattr__("bern_val_pos_"+str(cur_bsize))))
      if self.gpus>0:
        ll_pos = ll_pos.cuda()

      # neg samples
      neg_target = self.__getattr__("target_embed_"+str(t))(neg_targets[i])  
      #if self.use_meta:
      #  meta_idx = torch.multinomial(Variable(self.__getattr__("meta_neg_logits_"+str(cur_bsize))), self.ns)
      #  meta_feat = self.__getattr__("meta_embed")(meta_idx)
      #  meta_feat = self.__getattr__("mod_net")(meta_feat)
      #  neg_target = neg_target+meta_feat
      if self.use_meta:
        neg_target = neg_target+meta_feat.expand(cur_bsize, self.ns, self.K)
      neg_eta = torch.squeeze(torch.matmul(neg_target, pos_context_sum))
      y_neg = torch.distributions.Bernoulli(logits=neg_eta)
      ll_neg = y_neg.log_prob(Variable(self.__getattr__("bern_val_eval_neg_"+str(cur_bsize))))
      if self.gpus>0:
        ll_neg = ll_neg.cuda()
      ll_neg = ll_neg.sum(1)

      ll_total = ll_pos.squeeze()+ll_neg
      ll_total_master.append(ll_total)
    return ll_total_master
  
  def get_sparse_parameters(self):
    params = None
    for module in self.sparse_modules:
      cur_p = list(self.__getattr__(module).parameters())
      #if module.startswith("meta"):
      #  cur_p[0] = cur_p[0][1:]
      if not params:
        params = cur_p
      else:
        params += cur_p
    return params
  
  def get_dense_parameters(self):
    params = None
    for module in self.dense_modules:
      if not params:
        params = list(self.__getattr__(module).parameters())
      else:
        params += list(self.__getattr__(module).parameters())
    return params

  def count_trainable_parameters(self, params):
    model_parameters = filter(lambda p: p.requires_grad, params)
    return sum([np.prod(p.size()) for p in model_parameters])








