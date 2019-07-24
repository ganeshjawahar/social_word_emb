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

'''
# hyper-parameter search
# create data scripts
for context_size in [2, 4, 6, 8]:
  w_data_f.write("python "+code_folder+"/misc/preprocess.py --src_dir "+data_folder+" --dest_dir "+obj_folder+"/small_noshuffle_c"+str(context_size)+" --context_size "+str(context_size)+" --dry_run\n")

# create pretraining and dyntraining scripts
for dim in [100, 300, 500]:
  for lr in [1e-3, 0.01, 0.1, 1, 10]:
    for context_size in [2, 4, 6, 8]:
      w_premodel_f.write("python "+code_folder+"/premodel/pretrain.py --source_dir "+obj_folder+"/small_noshuffle_c"+str(context_size)+" --neg_dir "+data_folder+"/intrinsic/ll_neg --dest_dir lr"+str(lr)+"_dim"+str(dim)+" --n_epochs 1000 --n_iter 1 --ns 20 --gpus 0 --lr "+str(lr)+" --K "+str(dim)+" > out_pre_"+str(lr)+"_"+str(dim)+"_"+str(context_size)+"\n")
      init_embed = "--init_emb "+obj_folder+"/small_noshuffle_c"+str(context_size)+"/result/pretrain/lr"+str(lr)+"_dim"+str(dim)+"/embed_0.pkl"
      w_dynmodel_f.write("python "+code_folder+"/dynmodel/dyntrain.py "+init_embed+" --source_dir "+obj_folder+"/small_noshuffle_c"+str(context_size)+" --neg_dir "+data_folder+"/intrinsic/ll_neg --dest_dir lr"+str(lr)+"_dim"+str(dim)+" --n_epochs 1000 --n_iter 9 --ns 20 --gpus 0 --lr "+str(lr)+" --K "+str(dim)+" > out_dyn_"+str(lr)+"_"+str(dim)+"_"+str(context_size)+"\n")
'''

# sig search experiment
w_data_f.write("python "+code_folder+"/misc/preprocess.py --src_dir "+data_folder+" --dest_dir "+obj_folder+"/sig_search --dry_run\n")
for sig in range(10):
  sig = float(sig+1)
  w_premodel_f.write("python "+code_folder+"/premodel/pretrain.py --source_dir "+obj_folder+"/sig_search --neg_dir "+data_folder+"/intrinsic/ll_neg --dest_dir sig"+str(sig)+" --n_epochs 1000 --n_iter 1 --ns 20 --gpus 0 --sig "+str(sig)+" > "+obj_folder+"/sig_search/out_pre_"+str(sig)+"\n")
  init_embed = "--init_emb "+obj_folder+"/sig_search/result/pretrain/sig"+str(sig)+"/embed_0.pkl"
  w_dynmodel_f.write("python "+code_folder+"/dynmodel/dyntrain.py "+init_embed+" --source_dir "+obj_folder+"/sig_search --neg_dir "+data_folder+"/intrinsic/ll_neg --dest_dir sig"+str(sig)+" --n_epochs 1000 --n_iter 9 --ns 20 --gpus 0 --sig "+str(sig)+" > "+obj_folder+"/sig_search/out_dyn_"+str(sig)+"\n")

# create cleaning scripts
w_post_f.write("rm "+out_file+".data\n")
w_post_f.write("rm "+out_file+".premodel\n")
w_post_f.write("rm "+out_file+".dynmodel\n")
w_post_f.write("rm "+out_file+".post\n")

w_data_f.close()
w_premodel_f.close()
w_dynmodel_f.close()
w_post_f.close()
