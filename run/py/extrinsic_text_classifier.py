# Text classification using simple ML models
# text is represented by sum of word vectors

import sys
import os
import glob
import codecs
import argparse
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

'''
# code to create run scripts for all extrinsic text classificaiton tasks across meta models.
label_paths = ["extrinsic/indomain/sentiment_emoticon", "extrinsic/indomain/hashtag", "extrinsic/indomain/youtube_topic"]
root_embed_folder = "/home/ganesh/objects/dywoev/sept28_meta_search"
for label_path in label_paths:
  for embed_folder in glob.glob(root_embed_folder+"/*"):
    if os.path.isdir(embed_folder):
      res = 'python run/py/extrinsic_text_classifier.py --embed_dir %s/result/dyntrain/sample_run --label_path %s'%(embed_folder, label_path)
      print(res)
sys.exit(0)
'''

'''
# acl19 - stat. signific. results
data_root = '/home/gjawahar/data/dynword'
code_folder = '/home/gjawahar/projects/dywoev'
label_path = ['sentiment_emoticon', 'hashtag', 'youtube_topic', 'conversation_mention']
trained_models = '/home/gjawahar/data/dynword/trained_models'
obj_folder = '/home/gjawahar/objects/dynword/statsigni'
commands = []
for label in label_path:
  for model in glob.glob(trained_models+"/*"):
    idi = '%s_%s'%(label, model.split("/")[-1])
    command = 'python %s/run/py/extrinsic_text_classifier.py --root_dir %s --label_path %s --embed_dir %s --out_res %s/out_%s'%(code_folder, data_root, label, model, obj_folder, idi)
    commands.append(command)
print(len(commands))
# sys.exit(0)
scripts_dir = obj_folder + "/scripts"
side_dir = scripts_dir + "/side"
batch_size, bi = 1, 0
final_oar = open(scripts_dir + "/final_oar.sh", 'w')
final_oar.write("chmod 777 *\n")
final_oar.write("chmod 777 side/*\n")
for insi in range(0, len(commands), batch_size):
  oar_f = scripts_dir + "/oar_" + str(bi) + ".sh"
  master_f = scripts_dir + "/master_" + str(bi) + ".sh"
  oar_writer = open(oar_f, 'w')
  master_writer = open(master_f, 'w')
  for curi, cur_cmd in enumerate(commands[insi:insi+batch_size]):
    side_f = side_dir + "/"+str(insi)+"_"+str(curi)+ ".sh"
    side_writer = open(side_f, 'w')
    side_writer.write("source activate bert-wolf\n")
    side_writer.write(cur_cmd+"\n")
    side_writer.close()
    master_writer.write("bash "+side_f+"\n")
  oar_writer.write("bash "+master_f+"\n")
  bi += 1
  master_writer.close()
  oar_writer.close()
  final_oar.write("oarsub -l /core=5,walltime=72:00:00 ./oar_%d.sh\n"%(bi-1))
final_oar.close()
sys.exit(0)
'''

parser = argparse.ArgumentParser(description="Text classification using simple ML model")
parser.add_argument("--root_dir", type=str, default="/home/ganesh/data/sosweet/full", help="root directory containing data and labels")
parser.add_argument("--label_path", type=str, default="extrinsic/indomain/youtube_topic", help="path containing the text to be classified along with their labels")
parser.add_argument("--embed_dir", type=str, default="/home/ganesh/objects/dywoev/trial/result/dyntrain/run1", help="directory containing dynamic word embeddings to be evaluated")
parser.add_argument('--seed', type=int, default=123, help='seed value to be set manually')
parser.add_argument("--neural", action='store_true', help="use neural classifier?")
parser.add_argument("--out_res", type=str, default="/home/gjawahar/objects/dynword/out", help="?")
args = parser.parse_args()
print(args)

# load vocabulary
word2id, id2word = {}, {}
with codecs.open(os.path.join(args.root_dir, 'main', 'vocab_25K'), 'r', 'utf-8') as f:
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
  if 'target_0' in target_embeds:
    word_dim = target_embeds['target_0'].shape[1]
  else:
    word_dim = target_embeds['target'].shape[1]
    target_embed = target_embeds['target']
    target_embeds = {}
    for yi in range(5):
      target_embeds['target_%d'%yi] = target_embed

# create train, dev and test samples
year2id = {'2014':0, '2015':1, '2016':2, '2017':3, '2018':4}
def get_wordvector(word_str, year_str):
  if word_str not in word2id:
    return None
  year_id = year2id[year_str]
  if is_baseline:
    if word_str not in target_embeds[year_id]:
      return None
    return target_embeds[year_id][word_str]
  return target_embeds['target_%d'%year_id][word2id[word_str]]
def vectorize(tweet_content, month_info, default_allowed):
  year_str = month_info.split("-")[0]
  tweet_vec, num_hits = np.zeros(word_dim, dtype=np.float32), 0
  for word_str in tweet_content.split():
    word_vec = get_wordvector(word_str, year_str)
    if word_vec is not None:
      tweet_vec += word_vec
      num_hits += 1
  if not default_allowed and num_hits==0:
    return None
  return tweet_vec
def is_default_allowed(fname):
  return False if (fname == 'train.tsv') else True

train_X, train_Y_true, dev_X, dev_Y_true, test_X, test_Y_true = [], [], [], [], [], []
cat2id, id2cat = {}, {}
for month_folder in glob.glob(os.path.join(args.root_dir, args.label_path, '*')):
  month_info = month_folder.split("/")[-1]
  for split_file in glob.glob(os.path.join(month_folder, '*')):
    with codecs.open(split_file, 'r', 'utf-8') as f:
      fname = split_file.split("/")[-1]
      for line in f:
        tweet_id, tweet_cat, tweet_content = line.strip().split("\t")
        tweet_vec = vectorize(tweet_content, month_info, is_default_allowed(fname))
        if tweet_cat not in cat2id:
          cat2id[tweet_cat] = len(id2cat)
          id2cat[cat2id[tweet_cat]] = tweet_cat
        if fname == 'train.tsv':
          if tweet_vec is not None:
            train_X.append(tweet_vec)
            train_Y_true.append(cat2id[tweet_cat])
        elif fname == 'dev.tsv':
          assert(tweet_vec is not None)
          dev_X.append(tweet_vec)
          dev_Y_true.append(cat2id[tweet_cat])
        elif fname == 'test.tsv':
          assert(tweet_vec is not None)
          test_X.append(tweet_vec)
          test_Y_true.append(cat2id[tweet_cat])
#train_X, train_Y_true, dev_X, dev_Y_true, test_X, test_Y_true = train_X[0:200], train_Y_true[0:200], dev_X[0:200], dev_Y_true[0:200], test_X[0:200], test_Y_true[0:200]
train_X = np.array(train_X, dtype=np.float32)
dev_X = np.array(dev_X, dtype=np.float32)
test_X = np.array(test_X, dtype=np.float32)

# perform classification and testing
if not args.neural:
  regs = [2**t for t in range(-2, 4, 1)]
  scores = []
  for reg in regs:
    clf = LogisticRegression(C=reg, random_state=args.seed)
    clf.fit(train_X, train_Y_true)
    dev_Y_pred = clf.predict(dev_X)
    scores.append(f1_score(dev_Y_true, dev_Y_pred, average='micro'))
  optreg = regs[np.argmax(scores)]
  dev_fscore = np.max(scores)
  clf = LogisticRegression(C=optreg, random_state=args.seed)
  clf.fit(train_X, train_Y_true)
  test_Y_pred = clf.predict(test_X)
  writer = open(args.out_res, 'w')
  for y_true, y_pred in zip(test_Y_pred, test_Y_true):
    score = "100.0" if y_true == y_pred else "0.0"
    writer.write(score+"\n")
  writer.close()
  test_score_final = f1_score(test_Y_true, test_Y_pred, average='micro')
  print("best reg = %.2f; dev score = %.4f; test score = %.4f;"%(optreg, dev_fscore, test_score_final))
else:
  from senteval.tools.classifier import MLP
  #from classifier import MLP
  regs = [10**t for t in range(-5, -1)]
  nhids = [50] #[50, 100, 200]
  dropouts = [0.0, 0.1] #, 0.2]
  props, scores = [], []
  feat_dim, nclasses = word_dim, np.unique(train_Y_true).shape[0]
  train_Y_true, dev_Y_true, test_Y_true = np.array(train_Y_true), np.array(dev_Y_true), np.array(test_Y_true)
  for hid in nhids:
    for dropout in dropouts:
      for reg in regs:
        classifier_config = {'nhid': hid, 'optim': 'adam', 'batch_size': 64, 'tenacity': 5, 'epoch_size': 4, 'dropout': dropout}
        clf = MLP(classifier_config, inputdim=feat_dim, nclasses=nclasses, l2reg=reg, seed=args.seed, cudaEfficient=True)
        clf.fit(train_X, train_Y_true, validation_data=(dev_X, dev_Y_true))
        scores.append(round(100*clf.score(dev_X, dev_Y_true), 2))
        props.append([hid, dropout, reg])
  opt_prop = props[np.argmax(scores)]
  dev_acc = np.max(scores)
  classifier_config = {'nhid': opt_prop[0], 'optim': 'adam', 'batch_size': 64, 'tenacity': 5, 'epoch_size': 4, 'dropout': opt_prop[1]}
  clf = MLP(classifier_config, inputdim=feat_dim, nclasses=nclasses, l2reg=opt_prop[2], seed=args.seed, cudaEfficient=True)
  clf.fit(train_X, train_Y_true, validation_data=(dev_X, dev_Y_true))
  test_score_final = round(100*clf.score(test_X, test_Y_true), 2)
  print("best reg = %.2f; best nhid = %d; best dropout = %.2f; dev score = %.4f; test score = %.4f;"%(opt_prop[2], opt_prop[0], opt_prop[1], dev_acc, test_score_final))









