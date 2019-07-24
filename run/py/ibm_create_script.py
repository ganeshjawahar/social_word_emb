
import sys
import os

run_name = sys.argv[1]
job_range = [int(idx) for idx in sys.argv[2].split(",")]
gpu_ids = [int(idx) for idx in os.environ['CUDA_VISIBLE_DEVICES'].split(",")]
dry_run = (len(sys.argv) > 3)

meta_ids = ['user', 'user', 'user', 'user', 'user', 'user', 'user', 'user', 'tweet']
meta_types = ['embed', 'embed', 'category', 'category', 'category', 'embed', 'presence', 'category', 'category' ]
meta_paths = ['content/all_tweets_50.txt', 'content/geo_tweets_50.txt', 'dept/feat.txt', 'income/insee_feat.txt', 'income/iris_feat.txt', 'network/feat.txt', 'presence/feat.txt', 'region/feat.txt', 'yt_category/feat.txt'] 
compos = ['add', 'maxpool', 'gateword', 'gatemeta', 'bilinear']
label_paths = ["extrinsic/indomain/sentiment_emoticon", "extrinsic/indomain/hashtag", "extrinsic/indomain/youtube_topic"]

root_folder = os.getcwd()
data_folder = root_folder + "/data"
obj_folder = root_folder + "/objects/v1"
code_folder = root_folder + "/code/v1"
out_folder = obj_folder+"/out"
scripts_folder = obj_folder+"/scripts"
master_script_file = scripts_folder+('/run_%s_%d_%d_%d.sh'%(run_name, job_range[0], job_range[1], len(gpu_ids)))

commands_master = []

for compo in compos:
  for meta_id, meta_type, meta_path in zip(meta_ids, meta_types, meta_paths):
    run_prefix = '%s_%s_%s_%s'%(compo, meta_id, meta_type, meta_path.replace('/', '_').replace('.','/')[0:-4])
    dry_run_str = "" if not dry_run else " --dry_run"
    preprocess_cmd = "python "+code_folder+"/misc/meta_preprocess.py --src_dir "+data_folder+" --dest_dir "+obj_folder+"/"+run_prefix+" --meta_id "+meta_id+" --meta_type "+meta_type+" --meta_path "+meta_path+dry_run_str+" > "+out_folder+"/out_dat_"+run_prefix
    n_epochs = '1000' if dry_run else '10000'
    pretrain_cmd = "python "+code_folder+"/premodel/pretrain.py --source_dir "+obj_folder+"/"+run_prefix+" --neg_dir "+data_folder+"/intrinsic/ll_neg --dest_dir sample_run --n_epochs "+n_epochs+" --n_iter 1 --ns 20 --gpus 1 > "+out_folder+"/out_pre_"+run_prefix
    n_iter = '1' if dry_run else '9'
    init_embed = "--init_emb "+obj_folder+"/"+run_prefix+"/result/pretrain/sample_run/embed_0.pkl"
    dyntrain_cmd = "python "+code_folder+"/metamodel/metatrain.py "+init_embed+" --source_dir "+obj_folder+"/"+run_prefix+" --neg_dir "+data_folder+"/intrinsic/ll_neg --dest_dir sample_run --n_epochs "+n_epochs+" --n_iter "+n_iter+" --ns 20 --gpus 1 --use_meta --meta_compo "+compo+" > "+out_folder+"/out_meta_"+run_prefix
    cluster_cmd = "python "+code_folder+"/run/py/intrinsic_cluster_yao.py --label_dir "+data_folder+" --embed_dir "+obj_folder+"/"+run_prefix+"/result/dyntrain/sample_run > "+out_folder+"/out_cluster_yao_"+run_prefix
    commands = [preprocess_cmd, pretrain_cmd, dyntrain_cmd, cluster_cmd]
    if not dry_run:
      for label_path in label_paths:
        tc_cmd = "python "+code_folder+"/run/py/extrinsic_text_classifier.py --root_dir "+data_folder+" --embed_dir "+obj_folder+"/"+run_prefix+"/result/dyntrain/sample_run --label_path "+label_path+" > "+out_folder+"/out_"+label_path.split("/")[-1]+"_"+run_prefix
        commands.append(tc_cmd)
    commands_master.append(commands)

# self-attention
run_prefix = 'self_attn'
dry_run_str = "" if not dry_run else " --dry_run"
preprocess_cmd = "python "+code_folder+"/misc/combine_meta_process.py --src_dir "+data_folder+" --dest_dir "+obj_folder+"/"+run_prefix+dry_run_str+" > "+out_folder+"/out_dat_"+run_prefix
n_epochs = '1000' if dry_run else '10000'
pretrain_cmd = "python "+code_folder+"/premodel/pretrain.py --source_dir "+obj_folder+"/"+run_prefix+" --neg_dir "+data_folder+"/intrinsic/ll_neg --dest_dir sample_run --n_epochs "+n_epochs+" --n_iter 1 --ns 20 --gpus 1 > "+out_folder+"/out_pre_"+run_prefix
n_iter = '1' if dry_run else '9'
init_embed = "--init_emb "+obj_folder+"/"+run_prefix+"/result/pretrain/sample_run/embed_0.pkl"
dyntrain_cmd = "python "+code_folder+"/metamodel/metatrain.py "+init_embed+" --source_dir "+obj_folder+"/"+run_prefix+" --neg_dir "+data_folder+"/intrinsic/ll_neg --dest_dir sample_run --n_epochs "+n_epochs+" --n_iter "+n_iter+" --ns 20 --gpus 1 --use_meta > "+out_folder+"/out_meta_"+run_prefix
cluster_cmd = "python "+code_folder+"/run/py/intrinsic_cluster_yao.py --label_dir "+data_folder+" --embed_dir "+obj_folder+"/"+run_prefix+"/result/dyntrain/sample_run > "+out_folder+"/out_cluster_yao_"+run_prefix
commands = [preprocess_cmd, pretrain_cmd, dyntrain_cmd, cluster_cmd]
if not dry_run:
  for label_path in label_paths:
    tc_cmd = "python "+code_folder+"/run/py/extrinsic_text_classifier.py --root_dir "+data_folder+" --embed_dir "+obj_folder+"/"+run_prefix+"/result/dyntrain/sample_run --label_path "+label_path+" > "+out_folder+"/out_"+label_path.split("/")[-1]+"_"+run_prefix
    commands.append(tc_cmd)
commands_master.append(commands)

# context-attention
run_prefix = 'context_attn'
dry_run_str = "" if not dry_run else " --dry_run"
preprocess_cmd = "python "+code_folder+"/misc/combine_meta_process.py --src_dir "+data_folder+" --dest_dir "+obj_folder+"/"+run_prefix+dry_run_str+" > "+out_folder+"/out_dat_"+run_prefix
n_epochs = '1000' if dry_run else '10000'
pretrain_cmd = "python "+code_folder+"/premodel/pretrain.py --source_dir "+obj_folder+"/"+run_prefix+" --neg_dir "+data_folder+"/intrinsic/ll_neg --dest_dir sample_run --n_epochs "+n_epochs+" --n_iter 1 --ns 20 --gpus 1 > "+out_folder+"/out_pre_"+run_prefix
n_iter = '1' if dry_run else '9'
init_embed = "--init_emb "+obj_folder+"/"+run_prefix+"/result/pretrain/sample_run/embed_0.pkl"
dyntrain_cmd = "python "+code_folder+"/metamodel/metatrain.py "+init_embed+" --source_dir "+obj_folder+"/"+run_prefix+" --neg_dir "+data_folder+"/intrinsic/ll_neg --dest_dir sample_run --n_epochs "+n_epochs+" --n_iter "+n_iter+" --ns 20 --gpus 1 --use_meta --context_attn > "+out_folder+"/out_meta_"+run_prefix
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

