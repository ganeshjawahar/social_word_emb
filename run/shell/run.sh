#source activate dynword_pytorch36

#python misc/arxiv_process.py --src_dir /home/ganesh/data/arxiv_ML --dest_dir /home/ganesh/objects/dywoev/arxiv_ML
#CUDA_VISIBLE_DEVICES=0 python premodel/pretrain.py --source_dir /home/ganesh/objects/dywoev/arxiv_ML --n_epochs 10000 --n_iter 1 --ns 20 --gpus 1 --dest_dir run1
#CUDA_VISIBLE_DEVICES=0 python dynmodel/dyntrain.py --source_dir /home/ganesh/objects/dywoev/arxiv_ML --init_emb /home/ganesh/objects/dywoev/arxiv_ML/result/pretrain/run1/embed_0.pkl --n_iter 9 --ns 20 --n_epochs 10000 --gpus 1 --dest_dir run1

#dry run for yearly
#python misc/preprocess.py --dest_dir /home/ganesh/objects/dywoev/trial
#CUDA_VISIBLE_DEVICES=0 python premodel/pretrain.py --source_dir /home/ganesh/objects/dywoev/trial --n_epochs 10000 --n_iter 1 --ns 20 --gpus 1 --dest_dir run1
#CUDA_VISIBLE_DEVICES=0 python premodel/preeval.py --source_dir /home/ganesh/objects/dywoev/trial --n_epochs 1000 --ns 20 --gpus 1 --dest_dir run1
#CUDA_VISIBLE_DEVICES=0 python dynmodel/dyntrain.py --source_dir /home/ganesh/objects/dywoev/trial --init_emb /home/ganesh/objects/dywoev/trial/result/pretrain/run1/embed_0.pkl --n_iter 9 --ns 20 --n_epochs 10000 --gpus 1 --dest_dir run1
#CUDA_VISIBLE_DEVICES=0 python dynmodel/dyneval.py --source_dir /home/ganesh/objects/dywoev/trial --ns 20 --n_epochs 1000 --gpus 1 --dest_dir run1 --init_emb /home/ganesh/objects/dywoev/trial/result/pretrain/run1/embed_0.pkl

#full run with 0 pad
#python misc/preprocess.py --dest_dir /home/ganesh/objects/dywoev/zeropad
#CUDA_VISIBLE_DEVICES=0 python premodel/pretrain.py --source_dir /home/ganesh/objects/dywoev/zeropad --n_epochs 1000 --n_iter 1 --ns 20 --gpus 1 --dest_dir run1
#CUDA_VISIBLE_DEVICES=0 python dynmodel/dyntrain.py --source_dir /home/ganesh/objects/dywoev/zeropad --init_emb /home/ganesh/objects/dywoev/zeropad/result/pretrain/run1/embed_0.pkl --n_iter 4 --ns 20 --n_epochs 1000 --gpus 1 --dest_dir run1

#full run with pad token (can be used for trial run of dyntrain)
#python misc/preprocess.py --dest_dir /home/ganesh/objects/dywoev/padtoken --dry_run
#CUDA_VISIBLE_DEVICES=1 python premodel/pretrain.py --source_dir /home/ganesh/objects/dywoev/padtoken --n_epochs 1000 --n_iter 1 --ns 20 --gpus 1 --dest_dir run1
#CUDA_VISIBLE_DEVICES=0 python dynmodel/dyntrain.py --source_dir /home/ganesh/objects/dywoev/padtoken --init_emb /home/ganesh/objects/dywoev/padtoken/result/pretrain/run1/embed_0.pkl --n_iter 9 --ns 20 --n_epochs 1000 --gpus 1 --dest_dir run1

#with current train/dev/test proportion - dry_run
#python misc/preprocess.py --dest_dir /home/ganesh/objects/dywoev/aug1/zeropad --dry_run
#CUDA_VISIBLE_DEVICES=1 python premodel/pretrain.py --source_dir /home/ganesh/objects/dywoev/aug1/zeropad --n_epochs 1000 --n_iter 1 --ns 20 --gpus 1 --dest_dir run1
#CUDA_VISIBLE_DEVICES=1 python dynmodel/dyntrain.py --source_dir /home/ganesh/objects/dywoev/aug1/zeropad --init_emb /home/ganesh/objects/dywoev/aug1/zeropad/result/pretrain/run1/embed_0.pkl --n_iter 9 --ns 20 --n_epochs 1000 --gpus 1 --dest_dir run1

#metamodel trail run
#python misc/meta_preprocess.py --dest_dir /home/ganesh/objects/dywoev/aug6/meta_trial --meta_id user --meta_type embed --meta_path network/feat.txt
#CUDA_VISIBLE_DEVICES=1 python premodel/pretrain.py --source_dir /home/ganesh/objects/dywoev/aug6/meta_trial --n_epochs 10000 --n_iter 1 --ns 20 --gpus 1 --dest_dir run1
#CUDA_VISIBLE_DEVICES=1 python metamodel/metatrain.py --source_dir /home/ganesh/objects/dywoev/aug6/meta_trial --n_epochs 10000 --n_iter 1 --ns 20 --gpus 1 --dest_dir run1 --init_emb /home/ganesh/objects/dywoev/aug6/meta_trial/result/pretrain/run1/embed_0.pkl --use_meta

#metamodel trail category
#python misc/meta_preprocess.py --dest_dir /home/ganesh/objects/dywoev/oct10/meta_trial --dry_run --meta_id user --meta_type category --meta_path dept/feat.txt
#CUDA_VISIBLE_DEVICES=1 python premodel/pretrain.py --source_dir /home/ganesh/objects/dywoev/oct10/meta_trial --n_epochs 1000 --n_iter 1 --ns 20 --gpus 1 --dest_dir run1
#CUDA_VISIBLE_DEVICES=1 python metamodel/metatrain.py --source_dir /home/ganesh/objects/dywoev/oct10/meta_trial --n_epochs 1000 --n_iter 1 --ns 20 --gpus 1 --dest_dir run1 --init_emb None --use_meta

#metamodel trail presence
#python misc/meta_preprocess.py --dest_dir /home/ganesh/objects/dywoev/aug6/meta_trial --dry_run --meta_id user --meta_type presence --meta_path presence/feat.txt
#CUDA_VISIBLE_DEVICES=1 python premodel/pretrain.py --source_dir /home/ganesh/objects/dywoev/aug6/meta_trial --n_epochs 1000 --n_iter 1 --ns 20 --gpus 1 --dest_dir run1
#CUDA_VISIBLE_DEVICES=1 python metamodel/metatrain.py --source_dir /home/ganesh/objects/dywoev/aug6/meta_trial --n_epochs 1000 --n_iter 1 --ns 20 --gpus 1 --dest_dir run1 --init_emb /home/ganesh/objects/dywoev/aug6/meta_trial/result/pretrain/run1/embed_0.pkl --use_meta

#dynmodel check in alpaga (trial)
#python misc/preprocess.py --src_dir /home/alpaga/ganesh/data/sosweet/full --dest_dir /home/alpaga/ganesh/objects/dywoev/padtoken --dry_run
#CUDA_VISIBLE_DEVICES=1 python premodel/pretrain.py --source_dir /home/alpaga/ganesh/objects/dywoev/padtoken --neg_dir /home/alpaga/ganesh/data/sosweet/full/intrinsic/ll_neg --n_epochs 1000 --n_iter 1 --ns 20 --gpus 0 --dest_dir run1
#CUDA_VISIBLE_DEVICES=1 python dynmodel/dyntrain.py --source_dir /home/alpaga/ganesh/objects/dywoev/padtoken --neg_dir /home/alpaga/ganesh/data/sosweet/full/intrinsic/ll_neg --init_emb /home/alpaga/ganesh/objects/dywoev/padtoken/result/pretrain/run1/embed_0.pkl --n_iter 1 --ns 20 --n_epochs 1000 --gpus 0 --dest_dir run1

#metamodel => simulate dyntrain
#python misc/meta_preprocess.py --dest_dir /home/ganesh/objects/dywoev/aug21/meta_trial --dry_run --meta_id user --meta_type presence --meta_path presence/feat.txt
#CUDA_VISIBLE_DEVICES=0 python premodel/pretrain.py --source_dir /home/ganesh/objects/dywoev/aug21/meta_trial --n_epochs 1000 --n_iter 1 --ns 20 --gpus 1 --dest_dir run1
#CUDA_VISIBLE_DEVICES=1 python metamodel/metatrain.py --source_dir /home/ganesh/objects/dywoev/aug21/meta_trial --n_epochs 1000 --n_iter 9 --ns 20 --gpus 1 --dest_dir run1 --init_emb /home/ganesh/objects/dywoev/aug21/meta_trial/result/pretrain/run1/embed_0.pkl

#attn-meta model
#meta_ids = ['user', 'user', 'user', 'user', 'user', 'user', 'user', 'user', 'tweet']
#meta_types = ['embed', 'embed', 'category', 'category', 'category', 'embed', 'presence', 'category', 'category' ]
#meta_paths = ['content/all_tweets_50.txt', 'content/geo_tweets_50.txt', 'dept/feat.txt', 'income/insee_feat.txt', 'income/iris_feat.txt', 'network/feat.txt', 'presence/feat.txt', 'region/feat.txt', 'yt_category/feat.txt']
#python misc/combine_meta_process.py --dest_dir /home/ganesh/objects/dywoev/meta_attn --meta_ids user,user,user --meta_types embed,embed,category --meta_paths content/all_tweets_50.txt,content/geo_tweets_50.txt,dept/feat.txt --dry_run
CUDA_VISIBLE_DEVICES=1 python metamodel/metatrain.py --source_dir /home/ganesh/objects/dywoev/meta_attn --n_epochs 1000 --n_iter 1 --ns 20 --gpus 1 --dest_dir run1 --use_meta --attn_type self
CUDA_VISIBLE_DEVICES=1 python metamodel/metatrain.py --source_dir /home/ganesh/objects/dywoev/meta_attn --n_epochs 1000 --n_iter 1 --ns 20 --gpus 1 --dest_dir run1 --use_meta --attn_type context

#network feature - big run - slowness bug
#python misc/meta_preprocess.py --dest_dir /home/ganesh/objects/dywoev/oct22/net_bug --meta_id user --meta_type embed --meta_path network/feat.txt
#CUDA_VISIBLE_DEVICES=0 python metamodel/metatrain.py --source_dir /home/ganesh/objects/dywoev/oct22/net_bug --n_epochs 10000 --n_iter 1 --ns 20 --gpus 1 --dest_dir run1 --init_emb None --use_meta

# data space - problem
python misc/combine_meta_process.py --dest_dir /home/ganesh/objects/dywoev/dataprob --dry_run --data_dir /home/ganesh/data/sosweet/full/intrinsic/kulkarni_eval/syntactic/prob_0.2 --feat_dir /home/ganesh/data/sosweet/full
CUDA_VISIBLE_DEVICES=1 python premodel/pretrain.py --source_dir /home/ganesh/objects/dywoev/dataprob --neg_dir /home/ganesh/data/sosweet/full/intrinsic/ll_neg --n_epochs 1000 --n_iter 1 --ns 20 --gpus 1 --dest_dir run1 --skip_eval
CUDA_VISIBLE_DEVICES=1 python metamodel/metatrain.py --source_dir /home/ganesh/objects/dywoev/dataprob --neg_dir /home/ganesh/data/sosweet/full/intrinsic/ll_neg --n_epochs 100 --n_iter 1 --ns 20 --gpus 1 --dest_dir run1 --use_meta --attn_type context --meta_id user,user --meta_type presence,embed --meta_path presence/feat.txt,network/feat.txt --skip_eval

# check dyntrain
python misc/preprocess.py --dest_dir /home/ganesh/objects/dywoev/dataprob --dry_run --src_dir /home/ganesh/data/sosweet/full/intrinsic/kulkarni_eval/frequent/prob_0.2
CUDA_VISIBLE_DEVICES=1 python premodel/pretrain.py --source_dir /home/ganesh/objects/dywoev/dataprob --n_epochs 1000 --n_iter 1 --ns 20 --gpus 1 --dest_dir run1 --skip_eval
CUDA_VISIBLE_DEVICES=1 python dynmodel/dyntrain.py --source_dir /home/ganesh/objects/dywoev/dataprob --init_emb /home/ganesh/objects/dywoev/dataprob/result/pretrain/run1/embed_0.pkl --n_iter 1 --ns 20 --n_epochs 1000 --gpus 1 --dest_dir run1 --skip_eval


# rioc - baseline
# oarsub -q gpu -l /core=12,walltime=150:00:00 -p "host='gpu001'" -I
# source activate dywoev_finale
python misc/preprocess.py --dest_dir /scratch/gjawahar/projects/objects/dywoev/finale/no_meta --src_dir /scratch/gjawahar/projects/data/sosweet_lyon/tokenized/tweet_text/zip_stopwords_oui/splits/full/main --dry_run
CUDA_VISIBLE_DEVICES=0 python premodel/pretrain.py --source_dir /scratch/gjawahar/projects/objects/dywoev/finale/no_meta --n_epochs 1000 --n_iter 1 --ns 20 --gpus 1 --neg_dir /scratch/gjawahar/projects/data/sosweet_lyon/tokenized/tweet_text/zip_stopwords_oui/splits/full/intrinsic/ll_neg --dest_dir prebase
CUDA_VISIBLE_DEVICES=1 python dynmodel/dyntrain.py --source_dir /scratch/gjawahar/projects/objects/dywoev/finale/no_meta --init_emb /scratch/gjawahar/projects/objects/dywoev/finale/no_meta/result/pretrain/prebase/embed_0.pkl --n_iter 1 --ns 20 --n_epochs 1000 --gpus 1 --dest_dir run1 --skip_eval
# rioc - proposed
python misc/combine_meta_process.py --dest_dir /scratch/gjawahar/projects/objects/dywoev/finale/yes_meta --dry_run --data_dir /scratch/gjawahar/projects/data/sosweet_lyon/tokenized/tweet_text/zip_stopwords_oui/splits/full/main --feat_dir /scratch/gjawahar/projects/data/sosweet_lyon/tokenized/tweet_text/zip_stopwords_oui/splits/full
CUDA_VISIBLE_DEVICES=0 python premodel/pretrain.py --source_dir /scratch/gjawahar/projects/objects/dywoev/finale/yes_meta --n_epochs 1000 --n_iter 1 --ns 20 --gpus 1 --neg_dir /scratch/gjawahar/projects/data/sosweet_lyon/tokenized/tweet_text/zip_stopwords_oui/splits/full/intrinsic/ll_neg --dest_dir transfer
CUDA_VISIBLE_DEVICES=1 python metamodel/metatrain.py --source_dir /scratch/gjawahar/projects/objects/dywoev/finale/yes_meta --neg_dir /scratch/gjawahar/projects/data/sosweet_lyon/tokenized/tweet_text/zip_stopwords_oui/splits/full/intrinsic/ll_neg --n_epochs 1000 --n_iter 1 --ns 20 --gpus 1 --meta_ids user,user,user,user,user,user,user,user,tweet --meta_types embed,embed,category,category,category,embed,presence,category,category --meta_path content/all_tweets_50.txt,content/geo_tweets_50.txt,dept/feat.txt,income/insee_feat.txt,income/iris_feat.txt,network/feat.txt,presence/feat.txt,region/feat.txt,yt_category/feat.txt --init_emb /scratch/gjawahar/projects/objects/dywoev/finale/yes_meta/result/pretrain/transfer/embed_0.pkl --dest_dir naive --use_meta --attn_type naive 
CUDA_VISIBLE_DEVICES=1 python metamodel/metatrain.py --source_dir /scratch/gjawahar/projects/objects/dywoev/finale/yes_meta --neg_dir /scratch/gjawahar/projects/data/sosweet_lyon/tokenized/tweet_text/zip_stopwords_oui/splits/full/intrinsic/ll_neg --n_epochs 1000 --n_iter 1 --ns 20 --gpus 1 --meta_ids user,user,user,user,user,user,user,user,tweet --meta_types embed,embed,category,category,category,embed,presence,category,category --meta_path content/all_tweets_50.txt,content/geo_tweets_50.txt,dept/feat.txt,income/insee_feat.txt,income/iris_feat.txt,network/feat.txt,presence/feat.txt,region/feat.txt,yt_category/feat.txt --init_emb /scratch/gjawahar/projects/objects/dywoev/finale/yes_meta/result/pretrain/transfer/embed_0.pkl --dest_dir self --use_meta --attn_type self
CUDA_VISIBLE_DEVICES=1 python metamodel/metatrain.py --source_dir /scratch/gjawahar/projects/objects/dywoev/finale/yes_meta --neg_dir /scratch/gjawahar/projects/data/sosweet_lyon/tokenized/tweet_text/zip_stopwords_oui/splits/full/intrinsic/ll_neg --n_epochs 1000 --n_iter 1 --ns 20 --gpus 1 --meta_ids user,user,user,user,user,user,user,user,tweet --meta_types embed,embed,category,category,category,embed,presence,category,category --meta_path content/all_tweets_50.txt,content/geo_tweets_50.txt,dept/feat.txt,income/insee_feat.txt,income/iris_feat.txt,network/feat.txt,presence/feat.txt,region/feat.txt,yt_category/feat.txt --init_emb /scratch/gjawahar/projects/objects/dywoev/finale/yes_meta/result/pretrain/transfer/embed_0.pkl --dest_dir context --use_meta --attn_type context  
# runs excluding kulkarni_eval
python run/py/rioc_create_script_v4_step0.py naacl19 0,1 0,1 2 dry_run

# serpentes - proposed
python misc/combine_meta_process.py --dest_dir /home/ganesh/objects/dywoev/dataprob --dry_run --data_dir /home/ganesh/data/sosweet/full/main --feat_dir /home/ganesh/data/sosweet/full
CUDA_VISIBLE_DEVICES=1 python metamodel/metatrain.py --source_dir /home/ganesh/objects/dywoev/dataprob --neg_dir /home/ganesh/data/sosweet/full/intrinsic/ll_neg --n_epochs 1000 --n_iter 1 --ns 20 --gpus 1 --meta_ids user,user,user,user,user,user,user,user,tweet --meta_types embed,embed,category,category,category,embed,presence,category,category --meta_path content/all_tweets_50.txt,content/geo_tweets_50.txt,dept/feat.txt,income/insee_feat.txt,income/iris_feat.txt,network/feat.txt,presence/feat.txt,region/feat.txt,yt_category/feat.txt --dest_dir naive --use_meta --attn_type naive

# for hebrew
python misc/preprocess.py --dest_dir /home/ganesh/objects/dywoev/dataprob --src_dir /home/ganesh/data/hebrew/main/splits
CUDA_VISIBLE_DEVICES=1 python premodel/pretrain.py --source_dir /home/ganesh/objects/dywoev/dataprob --n_epochs 10000 --n_iter 1 --ns 20 --gpus 1 --dest_dir run1 --skip_eval
CUDA_VISIBLE_DEVICES=1 python dynmodel/dyntrain.py --source_dir /home/ganesh/objects/dywoev/dataprob --init_emb /home/ganesh/objects/dywoev/dataprob/result/pretrain/run1/embed_0.pkl --n_iter 9 --ns 20 --n_epochs 10000 --gpus 1 --dest_dir run1 --skip_eval













