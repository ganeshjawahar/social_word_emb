# tsne plot for word embeddings

import sys
import os
import glob
import codecs
import argparse
import numpy as np
import pickle
from sklearn.manifold import TSNE
# from plot_bok import bokey_plot

parser = argparse.ArgumentParser(description="Visualize TSNE")
parser.add_argument("--vocab_file", type=str, default="/Users/ganeshj/Desktop/dw2v/naacl19/vocab_25K", help="path to the file containing the vocabulary words")
parser.add_argument("--embed_dir", type=str, default="/Users/ganeshj/Desktop/dw2v/naacl19/naacl19-ibm/yes_meta/data_main/result/dyntrain/all_context", help="directory containing dynamic word embeddings to be evaluated")
#parser.add_argument('--time', type=str, default="0,4", help='time interval to examine')
parser.add_argument("--out_dir", type=str, default="/Users/ganesh/Desktop/sept28_meta_search/plot_dir", help="directory to output the images")
args = parser.parse_args()
print(args)

# load vocabulary
word2id, id2word = {}, {}
with codecs.open(args.vocab_file, 'r', 'utf-8') as f:
  for word_row in f:
    word_id, word_str, count = word_row.strip().split("\t")
    word_id, count = int(word_id), int(count)
    word2id[word_str] = word_id
    id2word[word_id] = word_str

# load word embeddings
is_baseline = os.path.exists(os.path.join(args.embed_dir, 'stats.pkl'))
if is_baseline:
  train_embed = pickle.load(open(os.path.join(args.embed_dir, 'stats.pkl'), 'rb'))
  time_period = train_embed["time"]
  context_size = train_embed["context_size"]
  word_dim = train_embed["word_dim"]
  context_embeds, target_embeds, mean_context_embeds, mean_target_embeds = {}, {}, {}, {}
  for cur_time, (context_embed, target_embed) in enumerate(zip(train_embed["contexts"], train_embed["targets"])):
    context_embeds[cur_time] = context_embed
    target_embeds[cur_time] = target_embed
    mean_context_embed, mean_target_embed, num_entries = np.zeros(word_dim), np.zeros(word_dim), 0.0
    for word in context_embed.keys():
      if word in word2id:
        mean_target_embed += target_embed[word]
        mean_context_embed += context_embed[word]
        num_entries += 1.0
    mean_target_embed /= num_entries
    mean_context_embed /= num_entries
    mean_target_embeds[cur_time] = mean_target_embed
    mean_context_embeds[cur_time] = mean_context_embed
else:
  model_id = max([ int(file.split('/')[-1].split('_')[-1].split('.')[0]) for file in glob.glob(os.path.join(args.embed_dir, 'embed_*.pkl'))])
  model_file = os.path.join(args.embed_dir, 'embed_%d.pkl'%model_id)
  print("loading model %s"%model_file)
  target_embeds = pickle.load(open(model_file, 'rb'))
  word_dim = target_embeds['target_0'].shape[1]

# find interested words
interesting_words = []
use_ben = False
if use_ben:
  ben_words = pickle.load(open("/Users/ganesh/Desktop/sept28_meta_search/sample_1.pkl", "rb"))
  for word in ben_words:
    word = word.lower()
    if word in word2id:
      interesting_words.append(word)
else:
  for idi in range(2000, 2301):
    word = id2word[idi]
    interesting_words.append(word)

print('%d interesting words are:'%len(interesting_words))
print(interesting_words)

'''
plot_dict = []
for ti in args.time.split(','):
  ti = int(ti)
  X, labels = [], []
  if is_baseline:
    for word in target_embeds[ti]:
      if word in interesting_words:
        X.append(target_embeds[ti][word])
        labels.append('%s_%d'%(word, ti))
  else:
    embeds = target_embeds['target_%d'%ti] #[0:100]
    for wi in range(len(id2word)):
      word = id2word[wi]
      if word in interesting_words:
        X.append(embeds[wi])
        labels.append('%s_%d'%(word, ti))
  X = np.array(X, dtype=np.float32)
  tsne = TSNE(n_iter=5000, perplexity=50, learning_rate=10, n_components=2)
  tr = tsne.fit_transform(X=X)
  plot_dict.append({"x":tr[:,0], "y":tr[:,1], "label":labels, "color":[len(plot_dict) for _ in range(len(tr))]})
if len(plot_dict)==1:
  bokey_plot(id_=None,mode="distinct",
           folder_bokey_default="/Users/ganeshj/Desktop/dw2v/naacl19/tsne_plots/",
           info=args.time,
           output=True,
           color_map_mode="divergent",
           dictionary_input={args.time.split(',')[0]: plot_dict[0]})
else:
  bokey_plot(id_=None,mode="single",
           folder_bokey_default="/Users/ganeshj/Desktop/dw2v/naacl19/tsne_plots/",
           info=args.time,
           output=True,
           color_map_mode="divergent",
           dictionary_input={args.time.split(',')[0]: plot_dict[0], args.time.split(',')[1]: plot_dict[1]})
'''

# plot using: https://github.com/kevinzakka/tsne-viz
# create tsne input
X, Y = [], []
for yi in range(5):
  for word in interesting_words:
    if is_baseline:
      if word in target_embeds[yi]:
        X.append(target_embeds[yi][word])
        Y.append(yi)
    else:
      idi = word2id[word]
      X.append(target_embeds['target_%d'%yi][idi])
      Y.append(yi)
X = np.array(X, dtype=np.float32)
Y = np.array(Y, dtype=np.int)
print(X.shape)
print(Y.shape)

# perform tsne
embeddings = TSNE(n_components=2, init='pca', verbose=2).fit_transform(X)
xx = embeddings[:, 0]
yy = embeddings[:, 1]

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.ticker import NullFormatter
fig = plt.figure()
ax = fig.add_subplot(111)
num_classes = 5
colors = cm.Spectral(np.linspace(0, 1, num_classes))
labels = np.arange(num_classes)
# plot the 2D data points
for i in range(num_classes):
  ax.scatter(xx[Y==i], yy[Y==i], color=colors[i], label=labels[i], s=10)
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')
plt.legend(loc='best', scatterpoints=1, fontsize=5)
plt.savefig(args.out_dir + '/' + args.embed_dir.split("/")[-1] +".pdf", format='pdf', dpi=600)
plt.show()















