# check for statistical significance using boostrap test

import sys
import numpy as np
import os
from tqdm import tqdm

'''
# bootstrap test scheduler
code_folder = '/home/gjawahar/projects/dywoev'
label_path = ['sentiment_emoticon', 'hashtag', 'youtube_topic', 'conversation_mention']
models = ['vanilla', 'hamilton', 'dbe', 'all_naive', 'all_self', 'all_context']
reference_models = [1, 2, 2, 2]
commands = []
obj_folder = '/home/gjawahar/objects/dynword/statsigni'
for label, ref_model in zip(label_path, reference_models):
  for model in models[3:]:
    resA = '%s/out_%s_%s'%(obj_folder, label, models[ref_model])
    resB = '%s/out_%s_%s'%(obj_folder, label, model)
    assert(os.path.exists(resA))
    assert(os.path.exists(resB))
    command = 'python %s/run/py/bootstrap_test.py %s %s 0.05 > %s/res_%s_%s_%s'%(code_folder, resA, resB, obj_folder, label, model, models[ref_model])
    commands.append(command)
scripts_dir = obj_folder + "/res_scripts"
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

#Bootstrap (copied from https://github.com/rtmdrr/testSignificanceNLP)
#Repeat R times: randomly create new samples from the data with repetitions, calculate delta(A,B).
# let r be the number of times that delta(A,B)<2*orig_delta(A,B). significance level: r/R
# This implementation follows the description in Berg-Kirkpatrick et al. (2012), 
# "An Empirical Investigation of Statistical Significance in NLP".
def Bootstrap(data_A, data_B, n, R):
  delta_orig = float(sum([x - y for x, y in zip(data_A, data_B)])) / n
  r = 0
  for x in tqdm(range(0, R)):
    temp_A = []
    temp_B = []
    samples = np.random.randint(0,n,n) #which samples to add to the subsample with repetitions
    for samp in samples:
      temp_A.append(data_A[samp])
      temp_B.append(data_B[samp])
    delta = float(sum([x - y for x, y in zip(temp_A, temp_B)])) / n
    if (delta < 2*delta_orig):
      r = r + 1
  pval = float(r)/(R)
  return pval

def read_res(fil):
  data = []
  with open(fil, 'r') as f:
    for line in f:
      data.append(float(line.strip()))
  return data

resA_file = sys.argv[1]
resB_file = sys.argv[2]
alpha = float(sys.argv[3])
data_A = read_res(resA_file)
data_B = read_res(resB_file)
assert len(data_A)==len(data_B)
assert len(data_A)>0
R = max(10000, int(len(data_A) * (1 / float(alpha))))
pval = Bootstrap(data_A, data_B, len(data_A), R)
if (float(pval) <= float(alpha)):
  print("Test result is significant with p-value: {}".format(pval))
else:
  print("Test result is not significant with p-value: {}".format(pval))












