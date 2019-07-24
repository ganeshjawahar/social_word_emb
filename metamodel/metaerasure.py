'''
model interpretation using representation erasure
ref: http://arxiv.org/abs/1612.08220
'''

import sys
import os
import argparse
import glob

from tqdm import tqdm
from metamodel import *
import numpy as np

import torch
import random
torch.manual_seed(123)
random.seed(123)

parser = argparse.ArgumentParser(description="interpretation script for dynamic word embeddings...")
parser.add_argument("--source_dir", type=str, default="/home/ganesh/objects/dywoev/trial", help="source dir")
parser.add_argument("--dest_dir", type=str, default="run1", help="dest dir")
parser.add_argument("--neg_dir", type=str, default="/home/ganesh/data/sosweet/full/intrinsic/ll_neg", help="dir where negative samples are present")
parser.add_argument('--K', type=int, default=100, help='Number of dimensions. Default is 100.')
parser.add_argument('--sig', type=float, default = 1.0, help='Noise on random walk for dynamic model. Default is 1.')
parser.add_argument('--n_iter', type=int, default = 1, help='Number of passes over the data. Default is 1.')
parser.add_argument('--n_epochs', type=int, default=1000, help='Number of epochs. Default is 10000.')
parser.add_argument('--ns', type=int, default=20, help='Number of negative samples. Default is 20.')
parser.add_argument("--lr", type=float, default=1e-3, help="initial learning rate")
parser.add_argument("--init_emb", type=str, default='/home/ganesh/objects/dywoev/monthly/result/pretrain/run1/embed_0.pkl', help="path to pre-trained word embeddings")
parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs to use. Default is 1.')
parser.add_argument("--sharing_time", type=str, default="dynamic", help="should the structures between adjacent time snapshots exist? static or dynamic")
parser.add_argument("--use_meta", action='store_true', help="use the meta features?")
parser.add_argument("--use_amortized", action='store_true', help="share statistical strength among members of group using amortization?")
parser.add_argument("--use_hierarchical", action='store_false', help="share statistical strength among members of group using hierarchical prior?")
parser.add_argument('--meta_compo', type=str, default='add', help='how to combine the meta embedding with the target word embedding? add, maxpool, gateword, gatemeta, bilinear, elemult')
parser.add_argument("--attn_type", type=str, help="Use (naive) average or (self) or (context)ual attention while combining the meta. embeddings.")
parser.add_argument("--meta_ids", type=str, default="user", help="user or tweet level meta information?")
parser.add_argument("--meta_types", type=str, default="embed", help="type of meta information. embed or category or presence?")
parser.add_argument("--meta_paths", type=str, default="network/feat.txt", help="relative path for the meta feature file")
parser.add_argument("--skip_eval", action='store_true', help="skip testing?")
args = parser.parse_args()

data = data(args)
model = model(args, data)
if args.gpus > 1:
  model = torch.nn.DataParallel(model)
if args.gpus > 0:
  model = model.cuda()

# load model
model_id = max([ int(file.split('/')[-1].split('_')[-1].split('.')[0]) for file in glob.glob(os.path.join(args.source_dir, 'result', 'dyntrain', args.dest_dir, '*'))])
model_id = str(model_id)
model_path = os.path.join(args.source_dir, 'result', 'dyntrain', args.dest_dir, 'embed_'+str(model_id)+'.pt')
model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
print('loaded model %s successfully'%(model_path))

def getModelModules(model):
  return model.module if args.gpus>1 else model

def evaluate(model, batch_gen, num_instances, batch_size, context_size, neg_batch_gen, meta_cache, erase_idx):
  #print('evaluating...')
  num_batches = np.amax(np.ceil((1.0*num_instances)/batch_size)).astype('int32')
  T = len(batch_gen)
  scores, num_processed = {}, [0]*T
  for t in range(T):
    scores[t] = np.array([0.0]*num_instances[t])
  for mi in range(len(data.meta_master)):
    if mi != erase_idx:
      model.__getattr__("meta_embed_%d"%mi).weight.data[:][:] = meta_cache[mi]
    else:
      model.__getattr__("meta_embed_%d"%mi).weight.data[:][:].fill_(0.0)
  # perform 0 padding
  for module in model.sparse_modules:
    if module.startswith('meta'):
      model.__getattr__(module).weight.data[0][:].fill_(0.0)
  for bi in tqdm(range(num_batches)):
    cur_context, cur_target, cur_meta, eval_range = [], [], [], []
    cur_neg_target = []
    for t in range(T):
      if num_processed[t] < num_instances[t]:
        eval_range.append(t)
        cur_batch = next(batch_gen[t])
        # prepare input
        for i, inp in enumerate(cur_batch):
          batch_target[t][i][0] = inp[0]
          batch_context[t][i] = torch.from_numpy(inp[1])
          if not data.multi_meta:
            batch_meta[t][i][0] = data.meta2id[inp[2]] if inp[2] in data.meta2id else 0
          else:
            for mi, mseq in enumerate(data.metaseq):
              batch_meta[t][mi][i][0] = data.meta_master[mi]['meta2id'][inp[2][mseq]] if inp[2][mseq] in data.meta_master[mi]['meta2id'] else 0
        cur_context.append(Variable(batch_context[t], requires_grad=False))
        cur_target.append(Variable(batch_target[t], requires_grad=False))
        if not data.multi_meta:
          cur_meta.append(Variable(batch_meta[t], requires_grad=False))
        else:
          cur_meta.append([Variable(batch_meta[t][mi], requires_grad=False) for mi in range(len(data.meta_master)) ])
        cur_neg_batch = next(neg_batch_gen[t])
        for i, inp in enumerate(cur_neg_batch):
          batch_neg_target[t][i] = torch.from_numpy(inp)
        cur_neg_target.append(Variable(batch_neg_target[t], requires_grad=False))
    assert(len(eval_range)!=0)
    ll_tot = model.eval(cur_context, cur_target, cur_neg_target, cur_meta, eval_range=eval_range)
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
batch_neg_target, batch_meta = [], []
for t in range(len(data.n_train)):
  batch_target.append(torch.LongTensor(data.n_train[t].item(), 1))
  batch_context.append(torch.LongTensor(data.n_train[t].item(), 2 * data.context_size))
  batch_neg_target.append(torch.LongTensor(data.n_train[t].item(), 20))
  if not data.multi_meta:
    batch_meta.append(torch.LongTensor(data.n_train[t].item(), 1))
  else:
    batch_meta.append([ torch.LongTensor(data.n_train[t].item(), 1) for mi in range(len(data.meta_master)) ])
  if args.gpus > 0:
    batch_target[t] = batch_target[t].cuda()
    batch_context[t] = batch_context[t].cuda()
    batch_neg_target[t] = batch_neg_target[t].cuda()
    if not data.multi_meta:
      batch_meta[t] = batch_meta[t].cuda()
    else:
      for mi in range(len(data.meta_master)):
        batch_meta[t][mi] = batch_meta[t][mi].cuda()

# create copy of all meta embeddings
meta_cache = {}
for mi in range(len(data.meta_master)):
  meta_cache[mi] = model.__getattr__("meta_embed_%d"%mi).weight.data.clone()

# erase one meta embedding at a time
meta_info = args.meta_paths.split(",")
for mi in range(len(data.meta_master)):
  print(meta_info[mi]+' = '+str(evaluate(getModelModules(model), data.test_batch, data.n_test, data.n_train, data.context_size, data.test_neg_batch, meta_cache, mi)))


