# dyn training step

import os
import sys
import glob

run_name = sys.argv[1]
job_range = [int(idx) for idx in sys.argv[2].split(",")]
gpu_ids = [int(idx) for idx in sys.argv[3].split(",")]
num_procs = int(sys.argv[4])
dry_run = (len(sys.argv) > 5)

meta_ids = ['user', 'user', 'user', 'user', 'user', 'user', 'user', 'user', 'tweet']
meta_types = ['embed', 'embed', 'category', 'category', 'category', 'embed', 'presence', 'category', 'category' ]
meta_paths = ['content/all_tweets_50.txt', 'content/geo_tweets_50.txt', 'dept/feat.txt', 'income/insee_feat.txt', 'income/iris_feat.txt', 'network/feat.txt', 'presence/feat.txt', 'region/feat.txt', 'yt_category/feat.txt'] 
compos = ['naive', 'self', 'context']

data_folder = "/scratch/gjawahar/projects/data/sosweet_lyon/tokenized/tweet_text/zip_stopwords_oui/splits/full"
obj_folder =  "/scratch/gjawahar/projects/objects/dywoev/"+run_name
code_folder = "/scratch/gjawahar/projects/dywoev"
out_folder = obj_folder+"/out"
scripts_folder = obj_folder+"/scripts"
master_script_file = scripts_folder+('/step1_%s_%d_%d_%d.sh'%(run_name, job_range[0], job_range[1], len(gpu_ids)))

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
meta_features = {}
meta_features['all'] = [interest, income, network, spatial, tcat, presence]
def get_feat(meta_feats):
  cur_ids, cur_types, cur_paths = [], [], []
  for mf in meta_feats:
    for mfi in mf:
      cur_ids.append(meta_ids[mfi])
      cur_types.append(meta_types[mfi])
      cur_paths.append(meta_paths[mfi])
  return ','.join(cur_ids), ','.join(cur_types), ','.join(cur_paths)

main_commands_master, synthetic_commands_master = [], []
data_dirs = glob.glob(data_folder + '/intrinsic/kulkarni_eval/*/*')  # [data_folder + '/main']
# baseline (pre-train)
meta_str = "no_meta"
for data_dir in data_dirs:
  #break
  sub_folder_name = data_dir.split("/")[-2] + "_" + data_dir.split("/")[-1]
  full_dest_folder_name = os.path.join(obj_folder, meta_str, sub_folder_name)
  n_iter = '1' if dry_run else '10'
  n_epochs = '1000' if dry_run else '10000'
  skip_eval_str = "" if 'kulkarni_eval' not in data_dir else " --skip_eval"
  cmd = "python "+code_folder+"/premodel/pretrain.py --source_dir "+full_dest_folder_name+" --neg_dir "+data_folder+"/intrinsic/ll_neg --dest_dir basepre --n_epochs "+n_epochs+" --n_iter "+n_iter+" --ns 20 --gpus 1"+skip_eval_str+" > "+out_folder+"/out_basepre_"+sub_folder_name
  if 'kulkarni_eval' in data_dir: 
    synthetic_commands_master.append(cmd)
  else:
    main_commands_master.append(cmd)
# baseline (dyn-train)
for data_dir in data_dirs:
  #break
  sub_folder_name = data_dir.split("/")[-2] + "_" + data_dir.split("/")[-1]
  full_dest_folder_name = os.path.join(obj_folder, meta_str, sub_folder_name)
  n_iter = '1' if dry_run else '9'
  n_epochs = '1000' if dry_run else '10000'
  skip_eval_str = "" if 'kulkarni_eval' not in data_dir else " --skip_eval"
  init_embed = "--init_emb "+full_dest_folder_name+"/result/pretrain/transfer/embed_0.pkl"
  cmd = "python "+code_folder+"/dynmodel/dyntrain.py "+init_embed+" --source_dir "+full_dest_folder_name+" --neg_dir "+data_folder+"/intrinsic/ll_neg --dest_dir basedyn --n_epochs "+n_epochs+" --n_iter "+n_iter+" --ns 20 --gpus 1"+skip_eval_str+" > "+out_folder+"/out_basedyn_"+sub_folder_name
  if 'kulkarni_eval' in data_dir: 
    synthetic_commands_master.append(cmd)
  else:
    main_commands_master.append(cmd)

# proposed (meta-train)
meta_str = "yes_meta"
for meta_feat in sorted(meta_features.keys()):
  break
  for compo in compos:
    run_prefix = "%s_%s"%(meta_feat, compo)
    cur_ids, cur_types, cur_paths = get_feat(meta_features[meta_feat])
    n_iter = '1' if dry_run else '9'
    n_epochs = '1000' if dry_run else '10000'
    cur_data_dirs = data_dirs if meta_feat == 'all' else [data_dirs[0]]
    for data_dir in cur_data_dirs:     
      sub_folder_name = data_dir.split("/")[-2] + "_" + data_dir.split("/")[-1]
      source_folder_name = os.path.join(obj_folder, meta_str, sub_folder_name)
      skip_eval_str = "" if 'kulkarni_eval' not in data_dir else " --skip_eval"
      init_embed = "--init_emb "+source_folder_name+"/result/pretrain/transfer/embed_0.pkl"
      cmd = "python "+code_folder+"/metamodel/metatrain.py "+init_embed+" --source_dir "+source_folder_name+" --neg_dir "+data_folder+"/intrinsic/ll_neg --dest_dir "+run_prefix+" --n_epochs "+n_epochs+" --n_iter "+n_iter+" --ns 20 --gpus 1"+skip_eval_str+" --use_meta --attn_type "+ compo + " --meta_ids "+cur_ids+" --meta_types "+ cur_types + " --meta_paths "+ cur_paths +" > "+out_folder+"/out_meta_"+sub_folder_name+"_"+run_prefix
      if 'kulkarni_eval' in data_dir: 
        synthetic_commands_master.append(cmd)
      else:
        main_commands_master.append(cmd)

commands_master = main_commands_master + synthetic_commands_master
print(len(commands_master))
cur_commands = commands_master[job_range[0]: job_range[1]+1]
num_gpu, cmi = 0, job_range[0]
w_master = open(master_script_file, 'w')
for command in cur_commands:
  w_side = open(scripts_folder + "/1_" + str(cmi) + ".sh", 'w')
  w_side.write("module load cuda80/toolkit/8.0.61\n")
  w_side.write("source activate dywoev_finale\n")
  cmd = "CUDA_VISIBLE_DEVICES="+str(gpu_ids[num_gpu])+" "+command
  w_side.write(cmd+"\n")
  w_side.close()
  num_gpu = (num_gpu + 1)%(len(gpu_ids))
  w_master.write("bash "+scripts_folder + "/1_" + str(cmi) + ".sh\n")
  cmi += 1
w_master.close()

oar_script_file = scripts_folder+('/step1_oar_%s_%d_%d_%d.sh'%(run_name, job_range[0], job_range[1], len(gpu_ids)))
w_master = open(oar_script_file, 'w')
w_master.write("bash %s/run/shell/giant_run.sh %s %d"%(code_folder, master_script_file, num_procs))
w_master.close()

print('run this file after chmoding :\n%s'%oar_script_file)







