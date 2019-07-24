'''
log-likelihood evaluation code for pretraining
'''

import sys
import os
import argparse
import glob

from tqdm import tqdm
from dynmodel import *
import numpy as np

import torch
import random
torch.manual_seed(123)
random.seed(123)

parser = argparse.ArgumentParser(description="main training script for dynamic word embeddings...")
parser.add_argument("--source_dir", type=str, default="/home/ganesh/objects/dywoev/trial", help="source dir")
parser.add_argument("--dest_dir", type=str, default="run1", help="dest dir")
parser.add_argument("--neg_dir", type=str, default="/scratch/gjawahar/projects/data/sosweet_lyon/tokenized/tweet_text/zip_stopwords_oui/splits/full_v2/intrinsic/ll_neg", help="dir where negative samples are present")
parser.add_argument('--K', type=int, default=100, help='Number of dimensions. Default is 100.')
parser.add_argument('--sig', type=float, default = 1.0, help='Noise on random walk for dynamic model. Default is 1.')
parser.add_argument('--n_epochs', type=int, default=1000, help='Number of epochs. Default is 10000.')
parser.add_argument('--ns', type=int, default=20, help='Number of negative samples. Default is 20.')
parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs to use. Default is 1.')
parser.add_argument("--init_emb", type=str, default='/home/ganesh/objects/dywoev/trial/result/pretrain/run1/embed_0.pkl', help="path to pre-trained word embeddings")
args = parser.parse_args()

data = data(args.source_dir, args.n_epochs, args.neg_dir)
model = model(args.K, args.sig, args.n_epochs, data.n_train, args.ns, data.unigram, data.context_size, args.init_emb, args.gpus)
if args.gpus > 1:
  model = torch.nn.DataParallel(model)
if args.gpus > 0:
  model = model.cuda()

# load model
model_id = max([ int(file.split('/')[-1].split('_')[-1].split('.')[0]) for file in glob.glob(os.path.join(args.source_dir, 'result', 'dyntrain', args.dest_dir, '*'))])
model_id = str(model_id)
model_path = os.path.join(args.source_dir, 'result', 'dyntrain', args.dest_dir, 'embed_'+str(model_id)+'.pt')
model.load_state_dict(torch.load(model_path)) # , map_location=lambda storage, loc: storage
print('loaded model %s successfully'%(model_path))

def getModelModules(model):
  return model.module if args.gpus>1 else model

def evaluate(model, batch_gen, num_instances, batch_size, context_size, neg_batch_gen):
  print('evaluating...')
  num_batches = np.amax(np.ceil(1.0*num_instances/batch_size)).astype('int32')
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
          batch_target[t][i][0] = inp[0]
          batch_context[t][i] = torch.from_numpy(inp[1])
        cur_context.append(Variable(batch_context[t], requires_grad=False))
        cur_target.append(Variable(batch_target[t], requires_grad=False))
        cur_neg_batch = next(neg_batch_gen[t])
        for i, inp in enumerate(cur_neg_batch):
          batch_neg_target[t][i] = torch.from_numpy(inp)
        cur_neg_target.append(Variable(batch_neg_target[t], requires_grad=False))
    assert(len(eval_range)!=0)
    ll_tot = model.eval(cur_context, cur_target, cur_neg_target, eval_range=eval_range)
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

batch_target, batch_context = [], []
batch_neg_target = []
for t in range(len(data.n_train)):
  batch_target.append(torch.LongTensor(data.n_train[t].item(), 1))
  batch_context.append(torch.LongTensor(data.n_train[t].item(), 2 * data.context_size))
  batch_neg_target.append(torch.LongTensor(data.n_train[t].item(), 20))
  if args.gpus > 0:
    batch_target[t] = batch_target[t].cuda()
    batch_context[t] = batch_context[t].cuda()
    batch_neg_target[t] = batch_neg_target[t].cuda()

#print('Dev-score  = '+str(evaluate(getModelModules(model), data.valid_batch, data.n_valid, data.n_train, data.context_size, data.valid_neg_batch)))
print('Test-score  = '+str(evaluate(getModelModules(model), data.test_batch, data.n_test, data.n_train, data.context_size, data.test_neg_batch)))


