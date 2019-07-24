import os
import sys
import glob

run_name = sys.argv[1]
job_range = [int(idx) for idx in sys.argv[2].split(",")]
gpu_ids = [int(idx) for idx in os.environ['CUDA_VISIBLE_DEVICES'].split(",")]
dry_run = (len(sys.argv) > 3)

meta_ids = ['user', 'user', 'user', 'user', 'user', 'user', 'user', 'user', 'tweet']
meta_types = ['embed', 'embed', 'category', 'category', 'category', 'embed', 'presence', 'category', 'category' ]
meta_paths = ['content/all_tweets_50.txt', 'content/geo_tweets_50.txt', 'dept/feat.txt', 'income/insee_feat.txt', 'income/iris_feat.txt', 'network/feat.txt', 'presence/feat.txt', 'region/feat.txt', 'yt_category/feat.txt'] 
label_paths = ["extrinsic/indomain/sentiment_emoticon", "extrinsic/indomain/hashtag", "extrinsic/indomain/youtube_topic"]
compos = ['naive', 'self', 'context']

root_folder = os.getcwd()
data_folder = root_folder + "/data"
obj_folder = root_folder + "/objects/v3"
code_folder = root_folder + "/code/v3"
out_folder = obj_folder+"/out"
scripts_folder = obj_folder+"/scripts"
master_script_file = scripts_folder+('/run_%s_%d_%d_%d.sh'%(run_name, job_range[0], job_range[1], len(gpu_ids)))

feature_groups = [['interest', [0, 1]], ['spatial', [2, 7]], ['income', [3, 4]], ['network', [5]], ['presence', [6]], ['tcat', [8]]]

from itertools import combinations, chain
allsubsets = lambda n: list(chain(*[combinations(range(n), ni) for ni in range(n+1)]))
meta_features = {}
for subset in allsubsets(len(feature_groups)):
  if len(subset)>0:
    key_str = ''
    vals = []
    for item in subset:
      feature_group = feature_groups[item]
      key_str += feature_group[0] + '_'
      vals.append(feature_group[1])
    key_str = key_str[0:-1]
    meta_features[key_str] = vals
def get_feat(meta_feats):
  cur_ids, cur_types, cur_paths = [], [], []
  for mf in meta_feats:
    for mfi in mf:
      cur_ids.append(meta_ids[mfi])
      cur_types.append(meta_types[mfi])
      cur_paths.append(meta_paths[mfi])
  return ','.join(cur_ids), ','.join(cur_types), ','.join(cur_paths)

commands_master = []
for meta_feat in sorted(meta_features.keys()):
  for compo in compos:
    if meta_feat in ['network', 'presence', 'tcat'] and compo != 'naive':
      continue
    run_prefix = "%s_%s"%(meta_feat, compo)
    dry_run_str = "" if not dry_run else " --dry_run"
    cur_ids, cur_types, cur_paths = get_feat(meta_features[meta_feat])
    preprocess_cmd = "python "+code_folder+"/misc/combine_meta_process.py --src_dir "+data_folder+" --dest_dir "+obj_folder+"/"+run_prefix+" --meta_ids "+cur_ids+" --meta_types "+cur_types+" --meta_paths "+cur_paths+dry_run_str+" > "+out_folder+"/out_dat_"+run_prefix
    n_epochs = '1000' if dry_run else '10000'
    pretrain_cmd = "python "+code_folder+"/premodel/pretrain.py --source_dir "+obj_folder+"/"+run_prefix+" --neg_dir "+data_folder+"/intrinsic/ll_neg --dest_dir sample_run --n_epochs "+n_epochs+" --n_iter 1 --ns 20 --gpus 1 > "+out_folder+"/out_pre_"+run_prefix
    n_iter = '1' if dry_run else '9'
    init_embed = "--init_emb "+obj_folder+"/"+run_prefix+"/result/pretrain/sample_run/embed_0.pkl"
    dyntrain_cmd = "python "+code_folder+"/metamodel/metatrain.py "+init_embed+" --source_dir "+obj_folder+"/"+run_prefix+" --neg_dir "+data_folder+"/intrinsic/ll_neg --dest_dir sample_run --n_epochs "+n_epochs+" --n_iter "+n_iter+" --ns 20 --gpus 1 --use_meta --attn_type "+ compo + " > "+out_folder+"/out_meta_"+run_prefix
    cluster_cmd = "python "+code_folder+"/run/py/intrinsic_cluster_yao.py --label_dir "+data_folder+" --embed_dir "+obj_folder+"/"+run_prefix+"/result/dyntrain/sample_run > "+out_folder+"/out_cluster_yao_"+run_prefix
    commands = [preprocess_cmd, pretrain_cmd, dyntrain_cmd, cluster_cmd]
    if not dry_run:
      for label_path in label_paths:
        tc_cmd = "python "+code_folder+"/run/py/extrinsic_text_classifier.py --root_dir "+data_folder+" --embed_dir "+obj_folder+"/"+run_prefix+"/result/dyntrain/sample_run --label_path "+label_path+" > "+out_folder+"/out_"+label_path.split("/")[-1]+"_"+run_prefix
        commands.append(tc_cmd)
    commands_master.append(commands)

cur_commands = commands_master[job_range[0]: job_range[1]+1]
num_gpu, cmi = 0, job_range[0]
w_master = open(master_script_file, 'w')
for commands in cur_commands:
  w_side = open(scripts_folder + "/" + str(cmi) + ".sh", 'w')
  for ci, cmd in enumerate(commands):
    if ci == 1 or ci == 2:
      cmd = "CUDA_VISIBLE_DEVICES="+str(gpu_ids[num_gpu])+" "+cmd
    w_side.write(cmd+"\n")
  w_side.close()
  num_gpu = (num_gpu + 1)%(len(gpu_ids))
  w_master.write("bash "+scripts_folder + "/" + str(cmi) + ".sh\n")
  cmi += 1
w_master.close()



