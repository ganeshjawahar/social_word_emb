# conduct a run over all proposed features

import os
import sys
import glob

data_folder = sys.argv[1]
obj_folder = sys.argv[2]
code_folder = sys.argv[3]
out_file = sys.argv[4]

w_data_f = open(out_file+".data", "w")
w_premodel_f = open(out_file+".premodel", "w")
w_dynmodel_f = open(out_file+".dynmodel", "w")
w_post_f = open(out_file+".post", "w")

meta_ids = ['user', 'user', 'user', 'user', 'user', 'user', 'user', 'user', 'tweet']
meta_types = ['embed', 'embed', 'category', 'category', 'category', 'embed', 'presence', 'category', 'category' ]
meta_paths = ['content/all_tweets_10.txt', 'content/all_tweets_50.txt', 'dept/feat.txt', 'income/insee_feat.txt', 'income/iris_feat.txt', 'network/feat.txt', 'presence/feat.txt', 'region/feat.txt', 'yt_category/feat.txt'] 

num_gpu = 0
for meta_id, meta_type, meta_path in zip(meta_ids, meta_types, meta_paths):
  num_gpu = 0
  w_data_f.write("python "+code_folder+"/misc/meta_preprocess.py --src_dir "+data_folder+" --dest_dir "+obj_folder+"/"+meta_id+"_"+meta_type+"_"+meta_path.replace('/','_').replace('..','/')[0:-4]+" --meta_id "+meta_id+" --meta_type "+meta_type+" --meta_path "+meta_path+" --dry_run > "+obj_folder+"/out_dat_"+meta_id+"_"+meta_type+"_"+meta_path.replace('/','_').replace('..','_')[0:-4]+"\n")
  w_premodel_f.write("CUDA_VISIBLE_DEVICES="+str(num_gpu)+" python "+code_folder+"/premodel/pretrain.py --source_dir "+obj_folder+"/"+meta_id+"_"+meta_type+"_"+meta_path.replace('/','_').replace('..','_')[0:-4]+" --neg_dir "+data_folder+"/intrinsic/ll_neg --dest_dir sample_run --n_epochs 1000 --n_iter 1 --ns 20 --gpus 1 > "+obj_folder+"/out_pre_"+meta_id+"_"+meta_type+"_"+meta_path.replace('/','_').replace('..','_')[0:-4]+"\n")
  init_embed = "--init_emb "+obj_folder+"/"+meta_id+"_"+meta_type+"_"+meta_path.replace('/','_').replace('..','_')[0:-4]+"/result/pretrain/sample_run/embed_0.pkl"
  w_dynmodel_f.write("CUDA_VISIBLE_DEVICES="+str(num_gpu)+" python "+code_folder+"/metamodel/metatrain.py "+init_embed+" --source_dir "+obj_folder+"/"+meta_id+"_"+meta_type+"_"+meta_path.replace('/','_')[0:-4]+" --neg_dir "+data_folder+"/intrinsic/ll_neg --dest_dir sample_run --n_epochs 1000 --n_iter 9 --ns 20 --gpus 1 --use_meta --meta_compo elemult > "+obj_folder+"/out_meta_"+meta_id+"_"+meta_type+"_"+meta_path.replace('/','_').replace('..','_')[0:-4]+"\n")
  num_gpu = (num_gpu+1)%2

# create cleaning scripts
w_post_f.write("rm "+out_file+".data\n")
w_post_f.write("rm "+out_file+".premodel\n")
w_post_f.write("rm "+out_file+".dynmodel\n")
w_post_f.write("rm "+out_file+".post\n")

w_data_f.close()
w_premodel_f.close()
w_dynmodel_f.close()
w_post_f.close()