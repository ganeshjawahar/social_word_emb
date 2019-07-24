'''
model attention visualization
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

def visualization_attn(model, batch_gen, num_instances, batch_size, context_size, neg_batch_gen, meta_info):
  num_batches = np.amax(np.ceil((1.0*num_instances)/batch_size)).astype('int32')
  T = len(batch_gen)
  num_processed = [0]*T
  # perform 0 padding
  for module in model.sparse_modules:
    if module.startswith('meta'):
      model.__getattr__(module).weight.data[0][:].fill_(0.0)
  vocab_attn, vocab_instances = {}, {}
  for ti in range(T):
    vocab_attn[ti] = np.zeros((model.L, 9), dtype=np.float64)
    vocab_instances[ti] = np.zeros((model.L, 9), dtype=np.float64)
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
    #ll_tot = model.eval(cur_context, cur_target, cur_neg_target, cur_meta, eval_range=eval_range)
    for i, t in enumerate(eval_range):
      cur_bsize = cur_context[i].data.size(0)
      # pos samples
      pos_context, pos_target = model.__getattr__("context_embed")(cur_context[i]), model.__getattr__("target_embed_"+str(t))(cur_target[i])
      pos_context_sum = torch.unsqueeze(torch.sum(pos_context, 1), 2)
      if model.attn_type != 'naive':
        attn_alphas, meta_embeds = None, None
        attention_weight = model.__getattr__("attention_weight")
        attention_bias = model.__getattr__("attention_bias")[0]
        for mi in range(model.num_meta):
          meta_embed = model.__getattr__("meta_embed_%d"%mi)(cur_meta[i][mi])
          meta_embed = model.__getattr__("proj_net_%d"%mi)(meta_embed)
          if model.attn_type == 'self':
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
        actual_bsize = min(num_instances[t], (bi+1) * batch_size[t]) - (bi * batch_size[t])
        num_processed[t] = num_processed[t] + actual_bsize
        cur_target_i = cur_target[i].squeeze().cpu().numpy()[0:actual_bsize]
        attn_alphas = attn_alphas.squeeze().detach().cpu().numpy()

        vocab_attn[i][cur_target_i] = vocab_attn[i][cur_target_i] + attn_alphas[0:actual_bsize]
        vocab_instances[i][cur_target_i] = vocab_instances[i][cur_target_i] + np.ones((actual_bsize, 9))
    # break
  for ti in range(T):
    vocab_attn[ti] = vocab_attn[ti] / vocab_instances[ti]
  pickle.dump(vocab_attn, open(os.path.join(args.source_dir, 'result', 'dyntrain', args.dest_dir, 'attn.pkl'), 'wb'))

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

# erase one meta embedding at a time
meta_info = args.meta_paths.split(",")
visualization_attn(getModelModules(model), data.test_batch, data.n_test, data.n_train, data.context_size, data.test_neg_batch, meta_info)


