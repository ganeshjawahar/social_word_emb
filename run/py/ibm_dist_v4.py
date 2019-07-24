import os
import sys
import glob

run_name = sys.argv[1]
num_gpus_per_jsub_call = int(sys.argv[2])
version_no = int(sys.argv[3])
num_procs = int(sys.argv[4])
cur_step = int(sys.argv[5]) # 0 - data collection and pretrain; 1 - pre-train 
dry_run = (len(sys.argv) > 6)

root_folder = os.getcwd()
obj_folder = root_folder + "/objects/v%d"%version_no
code_folder = root_folder + "/code/v%d"%version_no
code_folder = "/home/ganesh/projects/dywoev"

out_folder = obj_folder+"/out"
jb_folder = obj_folder+"/jb"
scripts_folder = obj_folder+"/scripts"
if not os.path.exists(out_folder):
  os.makedirs(out_folder)
if not os.path.exists(scripts_folder):
  os.makedirs(scripts_folder)
if not os.path.exists(jb_folder):
  os.makedirs(jb_folder)

f_finale_run = scripts_folder +"/%s_%d.sh"%(run_name, cur_step)
w = open(f_finale_run, 'w')
for i in range(0, num_procs, num_gpus_per_jsub_call):
  l = i
  r =  min(num_procs-1, (i + num_gpus_per_jsub_call - 1))
  master_script_file = scripts_folder+('/jbsub_%s_%d_%d.sh'%(run_name, l, r))
  w_master = open(master_script_file, 'w')
  if not dry_run:
    w_master.write("python %s/run/py/ibm_create_script_v%d_step%d.py %s %d,%d\n"%(code_folder, version_no, cur_step, run_name, l, r))
  else:
    w_master.write("python %s/run/py/ibm_create_script_v%d_step%d.py %s %d,%d dry_run\n"%(code_folder, version_no, cur_step, run_name, l, r))
  w_master.write("bash %s %s/step%d_%s_%d_%d_%d.sh %d"%(code_folder+"/run/shell/giant_run.sh", scripts_folder, cur_step, run_name, l, r, num_gpus_per_jsub_call, num_gpus_per_jsub_call))
  w_master.close()
  w.write("bash %s\n"%(master_script_file))
  #w.write("jbsub -q x86_7d -cores 1X2+%d -mem %dG -err %s -out %s bash %s\n"%(num_gpus_per_jsub_call, num_gpus_per_jsub_call*8, jb_folder+"/err_%d"%i, jb_folder+"/out_%d"%i, master_script_file))
w.close()

print("execute the following: ")
print("bash "+f_finale_run)
