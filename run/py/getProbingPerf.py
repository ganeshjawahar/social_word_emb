# gather the interpretation scores
# correlate with the downstream task

import os
import sys

# NMI, sentiment, hashtag, topic, conv
word2vec = [0.0338, 71.54, 37.32, 34.98, 70.04]
histwords = [0.0416, 73.69, 36.75, 36.85, 70.17]
dbe = [0.0651, 73, 41.83, 40.01, 70.98]
no_attn = [0.0575, 73.22, 42.11, 39.61, 71.21]
self_attn = [0.0614, 73.18, 42.19, 39.67, 71.1]
context_attn = [0.0683, 73.19, 41.88, 39.65, 71.15]
downstream_scores = [word2vec, histwords, dbe, no_attn, self_attn, context_attn]
print(downstream_scores)

interpret_outs = '/scratch/gjawahar/projects/objects/dywoev/naacl19-ibm/analysis/interpret_embed'
interpret_outs = 'interpret_embed'
int_tasks = ['sentlen', 'wc', 'treedepth', 'topconst', 'bshift', 'tense', 'subjnum', 'objnum', 'somo', 'coordinv']
method_names = ['baseline_vanilla', 'baseline_hamilton', 'no_meta_basedyn', 'yes_meta_all_naive', 'yes_meta_all_self', 'yes_meta_all_context']
probing_scores = []
res = ''
for method in method_names:
  meth_score = []
  for int_task in int_tasks:
    out_f = os.path.join(interpret_outs, 'out_%s_%s'%(method, int_task))
    score = None
    with open(out_f, 'r') as f:
      for line in f:
        content = line.strip()
        if content.startswith("best reg"):
          score = content.split()[-1][0:-1]
    meth_score.append(score)
    if score is None:
      print(out_f)
    assert(score is not None)
    #if score is None:
    #  score = '75.0'
    if score is not None:
      res += score + ","
  res += "\n"
  probing_scores.append(meth_score)
print(probing_scores)

'''
print(' & '.join(['']+int_tasks)+" \\\\")
for ps, probing_score in enumerate(probing_scores):
  print(' & '.join([method_names[ps]]+probing_score)+" \\\\")
'''

from scipy.stats import spearmanr
import numpy as np
import matplotlib.pyplot as pt
cmat = np.zeros((len(int_tasks), 5))
for ti in range(len(int_tasks)):
  for di in range(5):
    int_vals = []
    for i in range(len(probing_scores)):
      int_vals.append(float(probing_scores[i][ti]))
    down_vals = []
    for i in range(len(downstream_scores)):
      down_vals.append(downstream_scores[i][di])
    cmat[ti][di] = spearmanr(int_vals, down_vals)[0]
fig = pt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cmat)
fig.colorbar(cax)
int_tasks = ['SentLen', 'WC', 'TreeDepth', 'TopConst', 'BShift', 'Tense', 'SubjNum', 'ObjNum', 'SOMO', 'CoordInv']
down_tasks = ['SS', 'Senti', 'Htag', 'Topic', 'Conv']
ax.set_xticklabels(['']+down_tasks)

ax.set_yticks(np.arange(10), minor=False)
ax.set_yticklabels(int_tasks, minor=False)
pt.show()











