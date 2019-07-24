import gensim
import codecs
import os
import datetime
import numpy as np
import sys
import sklearn
from gensim_word2vec_procrustes_align import smart_procrustes_align_gensim, intersection_align_gensim
from numpy import dot
from numpy.linalg import norm
from tqdm import tqdm

def cos_sim(a, b):
  return dot(a, b)/(norm(a)*norm(b))

class Tweets(object):
  def __init__(self, file, folder):
    self.file = file
    self.folder = folder

  def __iter__(self):
    for f in self.file:
      with codecs.open(os.path.join(self.folder, f), 'r', 'utf-8') as f:
        i = 0
        for line in f:
          i += 1
          #if i > 1000:
          #  break
          yield line.strip().split()

def trainer(args, word2id, source_files_names, corpusSize, master_folder):
  def trim_rule(word, count, min_count):
    if word in word2id:
      return gensim.utils.RULE_KEEP
    return gensim.utils.RULE_DISCARD

  if args.logic == 0:
    return vanilla_word2vec(args, source_files_names, trim_rule, word2id, master_folder)
  elif args.logic == 1:
    return kim_word2vec(args, source_files_names, trim_rule, word2id, corpusSize, master_folder)
  elif args.logic == 2:
    return hamilton_word2vec(args, source_files_names, trim_rule, word2id, master_folder)

def vanilla_word2vec(args, source_files_names, trim_rule, word2id, master_folder):
  targets, contexts = [], []
  ti = -1
  for source_file in tqdm(source_files_names):
    ti += 1
    tweets = Tweets(source_file, os.path.join(args.data, 'train'))
    model = gensim.models.Word2Vec(sentences=tweets, sg=args.sg, size=args.size, window=args.window, alpha=args.alpha, min_alpha=args.min_alpha, seed=args.seed, min_count=args.min_count, max_vocab_size=args.max_vocab_size, workers=args.workers, hs=args.hs, negative=args.negative, iter=args.iter, trim_rule=trim_rule, compute_loss=args.compute_loss)
    
    target, context = {}, {}
    for word in model.wv.vocab.keys():
      target[word] = model.wv.get_vector(word)
      context[word] = model.syn1neg[model.wv.vocab[word].index]
    targets.append(target)
    contexts.append(context)

    model.save(os.path.join(master_folder, str(ti)+".model"))
  return targets, contexts

def kim_word2vec(args, source_files_names, trim_rule, word2id, corpusSize, master_folder):
  temp_fil_name = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
  ti = -1
  targets, contexts = [], []
  for source_file in tqdm(source_files_names):
    ti += 1
    tweets = Tweets(source_file, os.path.join(args.data, 'train'))
    model = None    
    if ti == 0:
      model = gensim.models.Word2Vec(sentences=tweets, sg=args.sg, size=args.size, window=args.window, alpha=args.alpha, min_alpha=args.min_alpha, seed=args.seed, min_count=args.min_count, max_vocab_size=args.max_vocab_size, workers=args.workers, hs=args.hs, negative=args.negative, iter=args.iter, trim_rule=trim_rule)
    else:  
      model = gensim.models.Word2Vec(sg=args.sg, size=args.size, window=args.window, alpha=args.alpha, min_alpha=args.min_alpha, seed=args.seed, min_count=args.min_count, max_vocab_size=args.max_vocab_size, hs=args.hs, negative=args.negative, iter=args.iter)
      model.build_vocab(tweets, trim_rule=trim_rule)

      # initialize with prev embedding
      model.intersect_word2vec_format(temp_fil_name)

      def compute_epoch_diff(prev_epoch_mat, cur_epoch_mat):
        score = 0.0
        for i in range(prev_epoch_mat.shape[0]):
          score += cos_sim(prev_epoch_mat[i], cur_epoch_mat[i])
        return score/prev_epoch_mat.shape[0]

      # training
      alpha = args.alpha
      diff = args.alpha - args.min_alpha
      dec_size = diff/args.iter
      cur_alpha = alpha
      #prev_epoch_mat = model.wv.syn0
      for it in range(args.iter):
        model.train(sentences=tweets, epochs=1, start_alpha=cur_alpha, end_alpha=cur_alpha-dec_size, queue_factor=args.workers, compute_loss=args.compute_loss, total_examples=corpusSize[ti])
        cur_alpha = cur_alpha-dec_size
        #cur_epoch_mat = model.wv.syn0
        #epoch_diff = compute_epoch_diff(prev_epoch_mat, cur_epoch_mat)
        #if epoch_diff >= args.threshold:
        #  print(file+" breaking at "+str(it)+" ("+str(epoch_diff))
        #  break

    model.wv.save_word2vec_format(temp_fil_name)
    model.save(os.path.join(master_folder, str(ti)+".model"))

    target, context = {}, {}
    for word in model.wv.vocab.keys():
      target[word] = model.wv.get_vector(word)
      context[word] = model.syn1neg[model.wv.vocab[word].index]
    targets.append(target)
    contexts.append(context)

  os.remove(temp_fil_name)
  return targets, contexts

def hamilton_word2vec(args, source_files_names, trim_rule, word2id, master_folder):
  prev_model = None
  ti = -1
  targets, contexts = [], []
  for source_file in tqdm(source_files_names):
    ti += 1
    tweets = Tweets(source_file, os.path.join(args.data, 'train'))
    model = gensim.models.Word2Vec(sentences=tweets, sg=args.sg, size=args.size, window=args.window, alpha=args.alpha, min_alpha=args.min_alpha, seed=args.seed, min_count=args.min_count, max_vocab_size=args.max_vocab_size, workers=args.workers, hs=args.hs, negative=args.negative, iter=args.iter, trim_rule=trim_rule, compute_loss=args.compute_loss)
    model.init_sims()
    if prev_model != None:
      model = smart_procrustes_align_gensim(prev_model, model)
    model.save(os.path.join(master_folder, str(ti)+".model"))
    prev_model = model

    target, context = {}, {}
    for word in model.wv.vocab.keys():
      target[word] = model.wv.get_vector(word)
      context[word] = model.syn1neg[model.wv.vocab[word].index]
    targets.append(target)
    contexts.append(context)
  return targets, contexts




