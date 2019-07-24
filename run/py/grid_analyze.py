# analyze the grid search results

import os
import sys
import glob

'''
#f_source = "/home/alpaga/ganesh/objects/dywoev/oarouts/smallrun_aug2"
f_source = "/Users/ganesh/deltable/smallrun_aug2"

# find aug1-best analysis here
results = []
for out_file in glob.glob(f_source+"/out_pre*"):
  dev_score, test_score = None, None
  with open(out_file, 'r') as f:
    for line in f:
      line = line.strip()
      if line.startswith("Dev-score"):
        dev_score = line.split()[3]
      if line.startswith("Test-score"):
        test_score = line.split()[3]
  assert(dev_score!=None)
  assert(test_score!=None)
  results.append([out_file.split("/")[-1][8:], float(dev_score), float(test_score)])
print('# results = %d'%(len(results)))

overall_best, best_100_2, best_100_8 = (-sys.maxsize - 1), (-sys.maxsize - 1), (-sys.maxsize - 1)
overall_best_ans, best_100_2_ans, best_100_8_ans = None, None, None
for result in results:
  params, dev_s, test_s = result
  if dev_s > overall_best:
    overall_best = dev_s
    overall_best_ans = result
  if params.endswith("100_2"):
    if dev_s > best_100_2:
      best_100_2 = dev_s
      best_100_2_ans = result
  if params.endswith("100_8"):
    if dev_s > best_100_8:
      best_100_8 = dev_s
      best_100_8_ans = result
print(best_100_2_ans)
print(best_100_8_ans)
print(overall_best_ans)
'''

'''
# find aug1-epoch analysis here
results = []
for out_file in glob.glob(f_source+"/out_dyn*"):
  best_dev_score, best_dev_epoch = (-sys.maxsize - 1), None
  with open(out_file, 'r') as f:
    for line in f:
      line = line.strip()
      if line.startswith("Dev-score"):
        dev_score = float(line.split()[3])
        if dev_score > best_dev_score:
          best_dev_score = dev_score
          best_dev_epoch = line.split()[1]
  assert(best_dev_epoch!=None)
  results.append([out_file.split("/")[-1][8:], best_dev_score, best_dev_epoch])
print('# results = %d'%(len(results)))

ep_map = {}
for result in results:
  params, dev_s, dev_e = result
  if dev_e not in ep_map:
    ep_map[dev_e] = 0
  ep_map[dev_e] += 1
print(ep_map)
'''

'''
meta_ids = ['user', 'user', 'user', 'user', 'user', 'user', 'user', 'user', 'tweet']
meta_types = ['embed', 'embed', 'category', 'category', 'category', 'embed', 'presence', 'category', 'category' ]
meta_paths = ['content/all_tweets_50.txt', 'content/geo_tweets_50.txt', 'dept/feat.txt', 'income/insee_feat.txt', 'income/iris_feat.txt', 'network/feat.txt', 'presence/feat.txt', 'region/feat.txt', 'yt_category/feat.txt'] 
compos = ['add', 'maxpool', 'gateword', 'gatemeta', 'bilinear']
label_paths = ["extrinsic/indomain/sentiment_emoticon", "extrinsic/indomain/hashtag", "extrinsic/indomain/youtube_topic"]
meta_reverse = {}
for compo in compos:
  for meta_id, meta_type, meta_path in zip(meta_ids, meta_types, meta_paths):
    run_prefix = '%s_%s_%s_%s'%(compo, meta_id, meta_type, meta_path.replace('/', '_').replace('.','/')[0:-4])
    small_run_prefix = '%s_%s'%(compo, meta_path.replace('/', '_').replace('.','/')[0:-4])
    meta_out = "out_meta_"+run_prefix
    meta_reverse[meta_out] = [small_run_prefix, 'll']
    cluster_out = "out_cluster_yao_"+run_prefix
    meta_reverse[cluster_out] = [small_run_prefix, 'cluster']
    for label_path in label_paths:
      label_out = "out_"+label_path.split("/")[-1]+"_"+run_prefix
      meta_reverse[label_out] = [small_run_prefix, label_path.split("/")[-1]]
for run_prefix in ['self_attn', 'context_attn']:
  meta_out = "out_meta_"+run_prefix
  meta_reverse[meta_out] = [run_prefix, 'll']
  cluster_out = "out_cluster_yao_"+run_prefix
  meta_reverse[cluster_out] = [run_prefix, 'cluster']
  for label_path in label_paths:
    label_out = "out_"+label_path.split("/")[-1]+"_"+run_prefix
    meta_reverse[label_out] = [run_prefix, label_path.split("/")[-1]]
'''
'''

# script_v2 analyze
meta_ids = ['user', 'user', 'user', 'user', 'user', 'user', 'user', 'user', 'tweet']
meta_types = ['embed', 'embed', 'category', 'category', 'category', 'embed', 'presence', 'category', 'category' ]
meta_paths = ['content/all_tweets_50.txt', 'content/geo_tweets_50.txt', 'dept/feat.txt', 'income/insee_feat.txt', 'income/iris_feat.txt', 'network/feat.txt', 'presence/feat.txt', 'region/feat.txt', 'yt_category/feat.txt'] 
compos = ['add', 'maxpool', 'gateword', 'gatemeta', 'bilinear']
label_paths = ["extrinsic/indomain/sentiment_emoticon", "extrinsic/indomain/hashtag", "extrinsic/indomain/youtube_topic"]

interest = [0, 1]
spatial = [2, 7]
income = [3, 4]
network = [5]
presence = [6]
tcat = [8]

meta_features = {}
meta_features['spatial'] = [spatial]
meta_features['income'] = [income]
meta_features['interest'] = [interest]
meta_features['spatial_income'] = [spatial, income]
meta_features['spatial_interest'] = [spatial, interest]
meta_features['income_interest'] = [income, interest]
meta_features['spatial_income_network'] = [spatial, income, network]
meta_features['spatial_interest_network'] = [spatial, interest, network]
meta_features['interest_income_network'] = [interest, income, network]
meta_features['interest_income_network_spatial'] = [interest, income, network, spatial]
meta_features['interest_income_network_spatial_presence'] = [interest, income, network, spatial, presence]
meta_features['interest_income_network_spatial_tcat'] = [interest, income, network, spatial, tcat]
meta_features['all'] = [interest, income, network, spatial, tcat, presence]

meta_reverse = {}
for meta_feat in meta_features:
  for ctx_attn in range(2):
    run_prefix = "%s_%d"%(meta_feat, ctx_attn)
    attn_str = 'context_attn' if ctx_attn==1 else 'self_attn'
    small_prefix = "%s_%s"%(meta_feat, attn_str)
    meta_reverse['out_meta_'+run_prefix] = [small_prefix, 'll']
    meta_reverse['out_cluster_yao_'+run_prefix] = [small_prefix, 'cluster']
    for label_path in label_paths:
      label_out = "out_"+label_path.split("/")[-1]+"_"+run_prefix
      meta_reverse[label_out] = [small_prefix, label_path.split("/")[-1]]

def get_featnscore(content):
  content = content.strip().split()[1].split("/")[-1]
  if content.startswith("out_dat") or content.startswith("out_pre"):
    return None, None
  assert content in meta_reverse
  return meta_reverse[content][0], meta_reverse[content][1]

def get_score(content):
  if content.startswith("best reg = "):
    return content.split()[-1][0:-1]
  if content.startswith("NMI = "):
    return content.split()[-1]
  if content.startswith("Test-score (iter=8) ="):
    return content.split()[-2] + " " + content.split()[-1]

f_source = "/Users/ganeshj/Desktop/tmp (5).txt"
meta_score = {}
feat_name, score_name = None, None
with open(f_source, 'r') as f:
  for line in f:
    content = line.strip()
    if len(content)==0:
      continue
    if content.startswith("==>"):
      feat_name, score_name = get_featnscore(content)
    else:
      if feat_name!=None:
        if content.startswith("best reg =") or content.startswith("NMI =") or content.startswith("Test-score (iter=8) ="):
          score = get_score(content)
          if feat_name not in meta_score:
            meta_score[feat_name] = {}
          meta_score[feat_name][score_name] = score
#print(meta_score)
feat_keys = ['ll', 'cluster', 'sentiment_emoticon', 'hashtag', 'youtube_topic']
for feat_name in sorted(meta_score.keys()):
  res = feat_name
  for score_name in feat_keys:
    res += ","+ meta_score[feat_name][score_name]
  print(res)
'''

'''
# collect synthetic eval results
out_folder = "/scratch/gjawahar/projects/objects/dywoev/naacl19-ccc/objects/v4/analysis/kulkarni_synthetic"
probs = [0.2, 0.4, 0.6, 0.8]
methods = ['no_meta_basedyn', 'yes_meta_all_naive', 'yes_meta_all_self', 'yes_meta_all_context']
for method in methods:
  res = ''
  for prob in probs:
    out_f = out_folder + '/out_'+method+'_syntactic_prob_'+str(prob)
    assert(os.path.exists(out_f))
    score = None
    with open(out_f, 'r') as f:
      for line in f:
        content = line.strip()
        if content.startswith("mrr = "):
          score = content.split(" = ")[1]
    assert(score is not None)
    res += score + ','
  res = res[0:-1]
  print(res)
'''

meta_features = []
meta_features.append('spatial')
meta_features.append('income')
meta_features.append('interest')
meta_features.append('spatial_income')
meta_features.append('spatial_interest')
meta_features.append('income_interest')
meta_features.append('spatial_income_network')
meta_features.append('spatial_interest_network')
meta_features.append('interest_income_network')
meta_features.append('interest_income_network_spatial')
meta_features.append('interest_income_network_spatial_presence')
meta_features.append('interest_income_network_spatial_tcat')
humfeats = []
humfeats.append('spatial')
humfeats.append('income')
humfeats.append('interest')
humfeats.append('spatial \& income')
humfeats.append('spatial \& interest')
humfeats.append('income \& interest')
humfeats.append('spatial \& income \& network')
humfeats.append('spatial \& interest \& network')
humfeats.append('interest \& income \& network')
humfeats.append('interest \& income \& network \& spatial')
humfeats.append('interest \& income \& network \& spatial \& knowledge')
humfeats.append('interest \& income \& network \& spatial \& topic')

results = []
# get ll results
ll_out_folder = '/scratch/gjawahar/projects/objects/dywoev/naacl19-ccc/objects/v4/out'
for compo in ['naive', 'self', 'context']:
  for meta_feat in meta_features:
    out_f = os.path.join(ll_out_folder, 'out_meta_data_main_'+meta_feat+"_"+compo)
    assert(os.path.exists(out_f))
    score = None
    with open(out_f, 'r') as f:
      for line in f:
        content = line.strip()
        if content.startswith("Test-score"):
          score = content.split()[-2]
    assert(score is not None)
    results.append([score])
# get cluster results
cluster_out_folder = '/scratch/gjawahar/projects/objects/dywoev/naacl19-ccc/objects/v4/analysis/cluster_yao'
idi = 0
for compo in ['naive', 'self', 'context']:
  for meta_feat in meta_features:
    out_f = os.path.join(cluster_out_folder, 'out_yes_meta_'+meta_feat+"_"+compo+"_data_main")
    assert(os.path.exists(out_f))
    score = None
    with open(out_f, 'r') as f:
      for line in f:
        content = line.strip()
        if content.startswith("NMI = "):
          score = content.split()[-1]
    assert(score is not None)
    results[idi].append(score)
    idi += 1
# get classification results
class_out_folder = '/scratch/gjawahar/projects/objects/dywoev/naacl19-ccc/objects/v4/analysis/text_classify'
idi = 0
for compo in ['naive', 'self', 'context']:
  for meta_feat in meta_features:
    for task in ['sentiment_emoticon', 'hashtag', 'youtube_topic', 'conversation_mention']:
      out_f = os.path.join(class_out_folder, 'out_yes_meta_'+meta_feat+"_"+compo+"_"+task+"_data_main")
      assert(os.path.exists(out_f))
      score = None
      with open(out_f, 'r') as f:
        for line in f:
          content = line.strip()
          if content.startswith("best reg = "):
            score = content.split()[-1][0:-1]
            score = float(score) * 100.0
            score = str(score)[0:5]
      assert(score is not None)
      results[idi].append(score)
    idi += 1
# write in latex
idi = 0
for result in results:
  res = humfeats[idi] + ' '
  for r in result:
    res += ' & ' + r
  print(res + ' \\\\ ')
  idi = (idi+1)%12













