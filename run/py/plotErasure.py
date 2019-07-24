# plot importance of each meta feature

import sys
import matplotlib.pyplot as pt
import numpy as np

self_score = -7.8598
context_score = -7.4293
naive_score = -8.1693
full_scores = [self_score, context_score]

self_erase_out = '/tmp/self_erasure'
context_erase_out = '/tmp/context_erasure'

rows = ['interest (all)', 'interest (geo)', 'dept.', 'income (insee)', 'income (iris)', 'network', 'knowledge', 'region', 'topic']
cols = ['self', 'context']
imp_mat = np.zeros((len(rows), len(cols)))
for fi, file in enumerate([self_erase_out, context_erase_out]):
  with open(file, 'r') as f:
    li = 0
    for line in f:
      content = line.strip()
      if content[-1] == ')':
        val = content.split()[-2]
        imp_mat[li][fi] = (full_scores[fi] - float(content[-2]))/full_scores[fi]
        li += 1
    assert(li==9)

fig = pt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(imp_mat)
fig.colorbar(cax)
ax.set_xticklabels(['']+cols)
ax.set_yticklabels(['']+rows)
pt.show()




