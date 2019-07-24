# data collection and pre-training step

import os
import sys
import glob

run_name = sys.argv[1]
job_range = [int(idx) for idx in sys.argv[2].split(",")]
#gpu_ids = [int(idx) for idx in os.environ['CUDA_VISIBLE_DEVICES'].split(",")]
gpu_ids = [0, 1]
dry_run = (len(sys.argv) > 3)

root_folder = os.getcwd()
#data_folder = root_folder + "/data"
data_folder = "/home/ganesh/data/sosweet/full"
obj_folder = root_folder + "/objects/v4"
#code_folder = root_folder + "/code/v4"
code_folder = "/home/ganesh/projects/dywoev"
out_folder = obj_folder+"/out"
scripts_folder = obj_folder+"/scripts"
master_script_file = scripts_folder+('/step0_%s_%d_%d_%d.sh'%(run_name, job_range[0], job_range[1], len(gpu_ids)))

commands_master = []
data_dirs = [data_folder + '/main'] + glob.glob(data_folder + '/intrinsic/kulkarni_eval/*/*')
for is_meta in range(2):
  meta_str = "no_meta" if is_meta==0 else "yes_meta"
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

cur_commands = commands_master[job_range[0]: job_range[1]+1]
num_gpu, cmi = 0, job_range[0]
w_master = open(master_script_file, 'w')
for commands in cur_commands:
  w_side = open(scripts_folder + "/0_" + str(cmi) + ".sh", 'w')
  for ci, cmd in enumerate(commands):
    if ci == 1:
      cmd = "CUDA_VISIBLE_DEVICES="+str(gpu_ids[num_gpu])+" "+cmd
    w_side.write(cmd+"\n")
  w_side.close()
  num_gpu = (num_gpu + 1)%(len(gpu_ids))
  w_master.write("bash "+scripts_folder + "/0_" + str(cmi) + ".sh\n")
  cmi += 1
w_master.close()





