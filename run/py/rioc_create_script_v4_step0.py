# data collection and pre-training step

import os
import sys
import glob

run_name = sys.argv[1]
job_range = [int(idx) for idx in sys.argv[2].split(",")]
gpu_ids = [int(idx) for idx in sys.argv[3].split(",")]
num_procs = int(sys.argv[4])
dry_run = (len(sys.argv) > 5)

data_folder = "/scratch/gjawahar/projects/data/sosweet_lyon/tokenized/tweet_text/zip_stopwords_oui/splits/full"
obj_folder =  "/scratch/gjawahar/projects/objects/dywoev/"+run_name
if not os.path.exists(obj_folder):
  os.makedirs(obj_folder)
code_folder = "/scratch/gjawahar/projects/dywoev"
out_folder = obj_folder+"/out"
if not os.path.exists(out_folder):
  os.makedirs(out_folder)
scripts_folder = obj_folder+"/scripts"
if not os.path.exists(scripts_folder):
  os.makedirs(scripts_folder)
master_script_file = scripts_folder+('/step0_%s_%d_%d_%d.sh'%(run_name, job_range[0], job_range[1], len(gpu_ids)))

commands_master = []
data_dirs = glob.glob(data_folder + '/intrinsic/kulkarni_eval/*/*') # [data_folder + '/main']
for is_meta in range(2):
  meta_str = "no_meta" if is_meta==0 else "yes_meta"
  if is_meta==1:
    continue
  for data_dir in data_dirs:
    sub_folder_name = data_dir.split("/")[-2] + "_" + data_dir.split("/")[-1]
    full_dest_folder_name = os.path.join(obj_folder, meta_str, sub_folder_name)
    dry_run_str = "" if not dry_run else " --dry_run"
    skip_eval_str = "" if 'kulkarni_eval' not in data_dir else " --skip_eval"
    n_epochs = '1000' if dry_run else '10000'
    preprocess_cmd = None
    if is_meta == 1:
      preprocess_cmd = "python "+code_folder+"/misc/combine_meta_process.py --data_dir "+data_dir+" --feat_dir "+data_folder+" --dest_dir "+full_dest_folder_name+dry_run_str+" > "+out_folder+"/out_dat_"+sub_folder_name
    else:
      preprocess_cmd = "python "+code_folder+"/misc/preprocess.py --src_dir "+data_dir+" --dest_dir "+full_dest_folder_name+dry_run_str+" > "+out_folder+"/out_dat_"+sub_folder_name
    pretrain_cmd = "python "+code_folder+"/premodel/pretrain.py --source_dir "+full_dest_folder_name+" --neg_dir "+data_folder+"/intrinsic/ll_neg --dest_dir transfer --n_epochs "+n_epochs+" --n_iter 1 --ns 20 --gpus 1"+skip_eval_str+" > "+out_folder+"/out_pre_"+sub_folder_name
    commands_master.append([preprocess_cmd, pretrain_cmd])
print(len(commands_master))

cur_commands = commands_master[job_range[0]: job_range[1]+1]
num_gpu, cmi = 0, job_range[0]
w_master = open(master_script_file, 'w')
for commands in cur_commands:
  w_side = open(scripts_folder + "/0_" + str(cmi) + ".sh", 'w')
  w_side.write("module load cuda80/toolkit/8.0.61\n")
  w_side.write("source activate dywoev_finale\n")
  for ci, cmd in enumerate(commands):
    if ci == 1:
      cmd = "CUDA_VISIBLE_DEVICES="+str(gpu_ids[num_gpu])+" "+cmd
    w_side.write(cmd+"\n")
  w_side.close()
  num_gpu = (num_gpu + 1)%(len(gpu_ids))
  w_master.write("bash "+scripts_folder + "/0_" + str(cmi) + ".sh\n")
  cmi += 1
w_master.close()

oar_script_file = scripts_folder+('/step0_oar_%s_%d_%d_%d.sh'%(run_name, job_range[0], job_range[1], len(gpu_ids)))
w_master = open(oar_script_file, 'w')
w_master.write("bash %s/run/shell/giant_run.sh %s %d"%(code_folder, master_script_file, num_procs))
w_master.close()

print('run this file after chmoding :\n%s'%oar_script_file)



