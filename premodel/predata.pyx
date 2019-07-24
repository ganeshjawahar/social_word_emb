import sys
import os
import pickle

import numpy as np
cimport numpy as np
import cython
np.random.seed(123)
import random
import pandas as pd

from libc.stdlib cimport malloc, free
from cpython cimport array

cdef class data:
  cdef public str src_path
  cdef public int n_epochs
  cdef char **source_files_names
  cdef public int N, n_train, n_valid, n_test, context_size
  #cdef public np.float64_t[:] counts
  #cdef public np.float64_t[:] unigram

  def __init__(self, str path, int n_epochs):
    self.src_path = path
    self.n_epochs = n_epochs

    # read data stats
    dat_stats = pickle.load(open(os.path.join(self.src_path, 'data', 'dat_stats.pkl'), 'rb'))
    self.source_files_names = self.to_cstring_array(dat_stats['source_files_names'])
    self.N, self.n_train, self.n_valid, self.n_test = 0, 0, 0, 0
    for i in range(len(dat_stats['num_tweets_map'])):
      self.N+=dat_stats['num_tweets_map'][i][2]
      self.n_valid+=dat_stats['num_tweets_map'][i][3]
      self.n_test+=dat_stats['num_tweets_map'][i][4]
    '''
    self.n_train = int(self.N/self.n_epochs)
    self.context_size = dat_stats['context_size']

    # load vocabulary
    df = pd.read_csv(os.path.join(self.src_path, 'data', 'unigram.txt'), delimiter='\t',header=None)
    self.counts = df[len(df.columns)-1].values
    counts = (1.0 * self.counts / self.N) ** (3.0 / 4)
    self.unigram = counts / self.N

    # data generator
    self.train_batch = self.batch_generator(self.n_train, 'train')
    self.valid_batch = self.batch_generator(self.n_train, 'valid') # batch size is fixed from train set
    self.test_batch = self.batch_generator(self.n_train, 'test') # batch size is fixed from train set
    '''
  
  cdef freemem(self):
    free(self.source_files_names)
  
  '''
  utility functions
  '''
  cdef char ** to_cstring_array(self, list_str):
    cdef char **ret = <char **>malloc(len(list_str) * sizeof(char *))
    for i in xrange(len(list_str)):
      ret[i] = list_str[i]
    return ret
  
  '''
  def batch_generator(self, batch_size, folder):
    f_idx = 0
    data = pickle.load(open(os.path.join(self.src_path, 'data', folder, self.source_files_names[f_idx]+'.pkl'), 'rb'))
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
          f_idx=(f_idx+1)%len(self.source_files_names)
          data = pickle.load(open(os.path.join(self.src_path, 'data', folder, self.source_files_names[f_idx]+'.pkl'), 'rb'))
          cur_idx = 0
          rand_idxs = np.random.permutation(len(data))
      assert(len(cur_batch)==batch_size)
      yield cur_batch
  '''


