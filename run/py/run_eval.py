# run evaluation script as a batch

import os
import sys
import glob

data_master_folder = "/scratch/gjawahar/projects/data/sosweet_lyon/tokenized/tweet_text/zip_stopwords_oui/splits/full"
model_embed_folder = "/scratch/gjawahar/projects/objects/dywoev/naacl19-ccc/objects/v4"
code_folder = "/scratch/gjawahar/projects/dywoev"
analysis_folder = model_embed_folder + "/analysis"
if not os.path.exists(analysis_folder):
  os.makedirs(analysis_folder)
scripts_folder = model_embed_folder + "/analysis/script"
if not os.path.exists(scripts_folder):
  os.makedirs(scripts_folder)
side_scripts_folder = model_embed_folder + "/analysis/script/side"
if not os.path.exists(side_scripts_folder):
  os.makedirs(side_scripts_folder)

# analysis to be done
cluster_yao = True
dubossar = False
text_classify = True
interpret_embed = False
synthetic_eval = True

commands_master = []

if cluster_yao:
  cluster_commands = []
  results_folder = analysis_folder + "/cluster_yao"
  if not os.path.exists(results_folder):
    os.makedirs(results_folder)
  for folder in ['yes_meta', 'no_meta']:
    meta_folder = model_embed_folder + "/" + folder
    if os.path.exists(meta_folder):
      for data_folder in glob.glob(meta_folder+"/*"):
        if "prob" in data_folder.split("/")[-1]:
          continue
        for run_root_folder in glob.glob(data_folder+"/result/*"):
          for run_child_folder in glob.glob(run_root_folder+"/*"):
            if run_child_folder.split("/")[-1] != 'transfer' and run_child_folder.split("/")[-1] != 'basepre':
              embed_files = glob.glob(run_child_folder+"/*")
              assert(len(embed_files)==1)
              cmd = 'python %s/run/py/intrinsic_cluster_yao.py --label_dir %s --embed_dir %s > %s/out_%s_%s_%s'%(code_folder, data_master_folder, run_child_folder, results_folder, folder, run_child_folder.split("/")[-1], data_folder.split("/")[-1])
              cluster_commands.append(cmd)
  commands_master.append(cluster_commands)

if dubossar:
  dubossar_commands = []
  results_folder = analysis_folder + "/dubossar"
  if not os.path.exists(results_folder):
    os.makedirs(results_folder)
  for folder in ['yes_meta', 'no_meta']:
    meta_folder = model_embed_folder + "/" + folder
    if os.path.exists(meta_folder):
      for data_folder in glob.glob(meta_folder+"/*"):
        for run_root_folder in glob.glob(data_folder+"/result/*"):
          for run_child_folder in glob.glob(run_root_folder+"/*"):
            if run_child_folder.split("/")[-1] != 'transfer':
              embed_files = glob.glob(run_child_folder+"/*")
              assert(len(embed_files)==1)
              cmd = 'python %s/run/py/intrinsic_dubossarsky.py --data_dir %s --embed_dir %s > %s/out_%s_%s'%(code_folder, data_master_folder, run_child_folder, results_folder, folder, run_child_folder.split("/")[-1])
              dubossar_commands.append(cmd)
  dubossar_commands = [dubossar_commands[-1]]
  commands_master.append(dubossar_commands)

if text_classify:
  tc_commands = []
  results_folder = analysis_folder + "/text_classify"
  if not os.path.exists(results_folder):
    os.makedirs(results_folder)
  datasets = ['sentiment_emoticon', 'hashtag', 'youtube_topic', 'conversation_mention']
  for folder in ['yes_meta', 'no_meta']:
    meta_folder = model_embed_folder + "/" + folder
    if os.path.exists(meta_folder):
      for data_folder in glob.glob(meta_folder+"/*"):
        if "prob" in data_folder.split("/")[-1]:
          continue
        for run_root_folder in glob.glob(data_folder+"/result/*"):
          for run_child_folder in glob.glob(run_root_folder+"/*"):
            if run_child_folder.split("/")[-1] != 'transfer' and run_child_folder.split("/")[-1] != 'basepre':
              embed_files = glob.glob(run_child_folder+"/*")
              assert(len(embed_files)==1)
              for dataset in datasets:
                cmd = 'python %s/run/py/extrinsic_text_classifier.py --root_dir %s --label_path extrinsic/indomain/%s --embed_dir %s > %s/out_%s_%s_%s_%s'%(code_folder, data_master_folder, dataset, run_child_folder, results_folder, folder, run_child_folder.split("/")[-1], dataset, data_folder.split("/")[-1])
                tc_commands.append(cmd)
  commands_master.append(tc_commands)

if interpret_embed:
  tc_commands = []
  results_folder = analysis_folder + "/interpret_embed"
  if not os.path.exists(results_folder):
    os.makedirs(results_folder)
  datasets = ['bshift', 'coordinv', 'objnum', 'sentlen', 'somo', 'subjnum', 'tense', 'topconst', 'treedepth', 'wc']
  '''
  for folder in ['yes_meta', 'no_meta']:
    meta_folder = model_embed_folder + "/" + folder
    if os.path.exists(meta_folder):
      for data_folder in glob.glob(meta_folder+"/*"):
        for run_root_folder in glob.glob(data_folder+"/result/*"):
          for run_child_folder in glob.glob(run_root_folder+"/*"):
            if run_child_folder.split("/")[-1] != 'transfer':
              embed_files = glob.glob(run_child_folder+"/*")
              assert(len(embed_files)==1)
              for dataset in datasets:
                cmd = 'python %s/run/py/extrinsic_text_classifier.py --neural --root_dir %s --label_path interpret/%s --embed_dir %s > %s/out_%s_%s_%s'%(code_folder, data_master_folder, dataset, run_child_folder, results_folder, folder, run_child_folder.split("/")[-1], dataset)
                tc_commands.append(cmd)
  '''
  # for baseline
  for embed_folder in ['/scratch/gjawahar/projects/objects/baselines/word2vec/vanilla', '/scratch/gjawahar/projects/objects/baselines/word2vec/hamilton']:
    for dataset in datasets:
      cmd = 'python %s/run/py/extrinsic_text_classifier.py --neural --root_dir %s --label_path interpret/%s --embed_dir %s > %s/out_%s_%s_%s'%(code_folder, data_master_folder, dataset, embed_folder, results_folder, 'baseline', embed_folder.split('/')[-1], dataset)
      tc_commands.append(cmd)
  commands_master.append(tc_commands)

if synthetic_eval:
  se_commands = []
  results_folder = analysis_folder + "/kulkarni_synthetic"
  if not os.path.exists(results_folder):
    os.makedirs(results_folder)
  for folder in ['yes_meta', 'no_meta']:
    meta_folder = model_embed_folder + "/" + folder
    if os.path.exists(meta_folder):
      for data_folder in glob.glob(meta_folder+"/*"):
        if "prob" not in data_folder.split("/")[-1]:
          continue
        for run_root_folder in glob.glob(data_folder+"/result/*"):
          for run_child_folder in glob.glob(run_root_folder+"/*"):
            if run_child_folder.split("/")[-1] != 'transfer' and run_child_folder.split("/")[-1] != 'basepre':
              embed_files = glob.glob(run_child_folder+"/*")
              assert(len(embed_files)==2)
              cmd = 'python %s/run/py/intrinsic_kulkarni_synthetic.py --data_dir %s --perturb_folder %s --embed_dir %s > %s/out_%s_%s_%s'%(code_folder, data_master_folder, data_folder.split("/")[-1].replace("frequent_","frequent/").replace("syntactic_","syntactic/"), run_child_folder, results_folder, folder, run_child_folder.split("/")[-1], data_folder.split("/")[-1])
              #cmd = 'python %s/1.py %s %s'%(code_folder, embed_files[0], embed_files[0].replace(".pkl", "_py27.pkl"))
              se_commands.append(cmd)
  commands_master.append(se_commands)

'''
for cmd_group in commands_master:
  for cmd in cmd_group:
    print(cmd)
  print(len(cmd_group))
sys.exit(0)
'''

bsize, cur_i, g_i = 40, 0, 0
for cmd_group in commands_master:
  for cmd in cmd_group:
    w_side = open(side_scripts_folder+"/%d_%d.sh"%(g_i, cur_i), 'w')
    if "intrinsic_kulkarni_synthetic.py" not in cmd:
      w_side.write("source activate dywoev_cpu\n")
    else:
      w_side.write("source activate py27\n")
    w_side.write(cmd+"\n")
    w_side.close()
    if cur_i == 0:
      w_master = open(scripts_folder+"/master_%d.sh"%g_i, 'w')
    w_master.write("bash %s/%d_%d.sh\n"%(side_scripts_folder, g_i, cur_i))
    cur_i += 1
    if cur_i == bsize:
      w_master.close()
      g_i += 1
      cur_i = 0
if cur_i!=0 and cur_i!=bsize:
  w_master.close()
  g_i += 1

for i in range(g_i):
  w_oar = open(scripts_folder+"/oar_%d.sh"%i, 'w')
  w_oar.write("bash %s/run/shell/giant_run.sh %s 40"%(code_folder, scripts_folder+"/master_%d.sh"%i))
  w_oar.close()
  print('oarsub -l /core=40,walltime=150:00:00 '+scripts_folder+"/oar_%d.sh"%i)

print('dont forget to chmod them!!!')



