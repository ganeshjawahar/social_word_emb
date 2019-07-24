'''
pretraining code for d-emb
'''

import sys
import os
import argparse
import datetime
import pickle
import math

from tqdm import tqdm
from premodel import *
#from predata import *

import torch
import numpy as np
import random
torch.manual_seed(123)
random.seed(123)

parser = argparse.ArgumentParser(description="pretraining script for dynamic word embeddings...")
parser.add_argument("--source_dir", type=str, default="/home/ganesh/objects/dywoev/trail", help="source dir")
parser.add_argument("--dest_dir", type=str, default="run1", help="dest dir")
parser.add_argument("--neg_dir", type=str, default="/home/ganesh/data/sosweet/full/intrinsic/ll_neg", help="dir where negative samples are present")
parser.add_argument('--K', type=int, default=100, help='Number of dimensions. Default is 100.')
parser.add_argument('--sig', type=float, default = 1.0, help='Noise on random walk for dynamic model. Default is 1.')
parser.add_argument('--n_iter', type=int, default = 1, help='Number of passes over the data. Default is 1.')
parser.add_argument('--n_epochs', type=int, default=10000, help='Number of epochs. Default is 10000.')
parser.add_argument('--ns', type=int, default=20, help='Number of negative samples. Default is 20.')
parser.add_argument("--lr", type=float, default=1e-3, help="initial learning rate")
parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs to use. Default is 1.')
parser.add_argument("--skip_eval", action='store_true', help="skip testing?")
args = parser.parse_args()

data = data(args.source_dir, args.n_epochs, args.neg_dir)
model = model(args.K, args.sig, args.n_epochs, data.n_train, args.ns, data.unigram, data.context_size, args.gpus)
if args.gpus > 1:
  model = torch.nn.DataParallel(model)
if args.gpus > 0:
  model = model.cuda()
optimizer = torch.optim.SparseAdam(model.parameters(), lr=args.lr)

res_folder_path = os.path.join(args.source_dir, 'result', 'pretrain', args.dest_dir)
if not os.path.exists(res_folder_path):
  os.makedirs(res_folder_path)
print('storing results in '+res_folder_path)

def getModelModules(model):
  return model.module if args.gpus>1 else model

def evaluate(model, batch_gen, num_instances, batch_size, context_size, neg_batch_gen):
  num_batches = int(math.ceil((1.0*num_instances.sum())/batch_size))
  scores = np.array([0.0]*num_instances.sum())
  print('evaluating...')
  for bi in tqdm(range(num_batches)):
    cur_batch = next(batch_gen)
    # prepare input
    for i, inp in enumerate(cur_batch):
      batch_target[i][0] = inp[0]
      batch_context[i] = torch.from_numpy(inp[1])
    cur_neg_batch = next(neg_batch_gen)
    for i, inp in enumerate(cur_neg_batch):
      batch_neg_target[i] = torch.from_numpy(inp)
    ll_tot = model.eval(Variable(batch_context, requires_grad=False), Variable(batch_target, requires_grad=False), Variable(batch_neg_target, requires_grad=False))
    if bi == num_batches-1:
      # last batch
      cur_bsize = num_instances.sum() - (bi * batch_size)
      scores[bi*batch_size: (bi*batch_size)+cur_bsize] = ll_tot.data.cpu().numpy()[0:cur_bsize]
    else:
      scores[bi*batch_size: (bi+1)*batch_size] = ll_tot.data.cpu().numpy()
  return "%.4f (%.4f)"%(scores.mean(),scores.std())

batch_target, batch_context = torch.LongTensor(data.n_train, 1), torch.LongTensor(data.n_train, 2 * data.context_size)
batch_neg_target = torch.LongTensor(data.n_train, 20)
if args.gpus > 0:
  batch_target = batch_target.cuda()
  batch_context = batch_context.cuda()
  batch_neg_target = batch_neg_target.cuda()

#print('Dev-score (before-train) = '+evaluate(getModelModules(model), data.valid_batch, data.n_valid, data.n_train, data.context_size, data.valid_neg_batch))
#print('Test-score (before-train) = '+evaluate(getModelModules(model), data.test_batch, data.n_test, data.n_train, data.context_size, data.test_neg_batch))

print('training...')
for iteration in range(args.n_iter):
  print('iter '+str(iteration)+' ...')
  cost = 0
  for epoch in tqdm(range(args.n_epochs)):
    cur_batch = next(data.train_batch)
    
    # prepare input
    for i, inp in enumerate(cur_batch):
      batch_target[i][0] = inp[0]
      batch_context[i] = torch.from_numpy(inp[1])
      
    loss = model(Variable(batch_context, requires_grad=False), Variable(batch_target, requires_grad=False))
    loss = loss.sum(0)
    cost += loss.data.item()
    assert(math.isnan(cost)==False)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  cost /= args.n_epochs
  print('Loss (iter=%d) = %.4f'%(iteration, cost))
  if iteration == args.n_iter-1:
    if not args.skip_eval:
      print('Dev-score (iter='+str(iteration)+') = '+evaluate(getModelModules(model), data.valid_batch, data.n_valid, data.n_train, data.context_size, data.valid_neg_batch))
      print('Test-score (iter='+str(iteration)+') = '+evaluate(getModelModules(model), data.test_batch, data.n_test, data.n_train, data.context_size, data.test_neg_batch))

    # save embeddings
    print('saving iteration '+str(iteration))
    save_obj = {}
    save_obj['context'] = getModelModules(model).context_embed.weight.data.cpu().numpy()
    save_obj['target'] = getModelModules(model).target_embed.weight.data.cpu().numpy()
    pickle.dump(save_obj, open(os.path.join(res_folder_path, 'embed_'+str(iteration)+'.pkl') ,'wb'))

  #torch.save(getModelModules(model).state_dict(), os.path.join(res_folder_path, 'embed_'+str(iteration)+'.pt'))


  

