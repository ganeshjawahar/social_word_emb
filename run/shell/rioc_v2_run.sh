#!/bin/bash
# run on full v2 data

source activate lordganesh

if [ $1 == "dbe" ] ; then
# baseline model
python misc/preprocess.py --dest_dir /scratch/gjawahar/objects/dywoev/full_v2/dbe --src_dir /scratch/gjawahar/projects/data/sosweet_lyon/tokenized/tweet_text/zip_stopwords_oui/splits/full_v2/main
CUDA_VISIBLE_DEVICES=0 python premodel/pretrain.py --source_dir /scratch/gjawahar/objects/dywoev/full_v2/dbe --n_epochs 10000 --n_iter 1 --ns 20 --gpus 1 --dest_dir run1 --skip_eval
CUDA_VISIBLE_DEVICES=0 python dynmodel/dyntrain.py --source_dir /scratch/gjawahar/objects/dywoev/full_v2/dbe --init_emb /scratch/gjawahar/objects/dywoev/full_v2/dbe/result/pretrain/run1/embed_0.pkl --n_iter 9 --ns 20 --n_epochs 10000 --gpus 1 --dest_dir dbe --skip_eval
CUDA_VISIBLE_DEVICES=0 python dynmodel/dyneval.py --source_dir /scratch/gjawahar/objects/dywoev/full_v2/dbe --ns 20 --n_epochs 10000 --gpus 1 --dest_dir dbe --init_emb /scratch/gjawahar/objects/dywoev/full_v2/dbe/result/pretrain/run1/embed_0.pkl
fi

if [ $1 == "noattn" ] ; then
# no attention model
python misc/combine_meta_process.py --dest_dir /scratch/gjawahar/objects/dywoev/full_v2/noattn --data_dir /scratch/gjawahar/projects/data/sosweet_lyon/tokenized/tweet_text/zip_stopwords_oui/splits/full_v2/main --feat_dir /scratch/gjawahar/projects/data/sosweet_lyon/tokenized/tweet_text/zip_stopwords_oui/splits/full_v2
CUDA_VISIBLE_DEVICES=1 python premodel/pretrain.py --source_dir /scratch/gjawahar/objects/dywoev/full_v2/noattn --n_epochs 10000 --n_iter 1 --ns 20 --gpus 1 --neg_dir /scratch/gjawahar/projects/data/sosweet_lyon/tokenized/tweet_text/zip_stopwords_oui/splits/full_v2/intrinsic/ll_neg --dest_dir run1 --skip_eval
CUDA_VISIBLE_DEVICES=1 python metamodel/metatrain.py --source_dir /scratch/gjawahar/objects/dywoev/full_v2/noattn --neg_dir /scratch/gjawahar/projects/data/sosweet_lyon/tokenized/tweet_text/zip_stopwords_oui/splits/full_v2/intrinsic/ll_neg --n_epochs 10000 --n_iter 9 --ns 20 --gpus 1 --meta_ids user,user,user,user,user,user,user,user,tweet --meta_types embed,embed,category,category,category,embed,presence,category,category --meta_path content/all_tweets_50.txt,content/geo_tweets_50.txt,dept/feat.txt,income/insee_feat.txt,income/iris_feat.txt,network/feat.txt,presence/feat.txt,region/feat.txt,yt_category/feat.txt --init_emb /scratch/gjawahar/objects/dywoev/full_v2/noattn/result/pretrain/run1/embed_0.pkl --dest_dir naive --use_meta --attn_type naive --skip_eval
CUDA_VISIBLE_DEVICES=1 python metamodel/metaeval.py --source_dir /scratch/gjawahar/objects/dywoev/full_v2/noattn --neg_dir /scratch/gjawahar/projects/data/sosweet_lyon/tokenized/tweet_text/zip_stopwords_oui/splits/full_v2/intrinsic/ll_neg --n_epochs 10000 --n_iter 1 --ns 20 --gpus 1 --meta_ids user,user,user,user,user,user,user,user,tweet --meta_types embed,embed,category,category,category,embed,presence,category,category --meta_path content/all_tweets_50.txt,content/geo_tweets_50.txt,dept/feat.txt,income/insee_feat.txt,income/iris_feat.txt,network/feat.txt,presence/feat.txt,region/feat.txt,yt_category/feat.txt --init_emb /scratch/gjawahar/objects/dywoev/full_v2/noattn/result/pretrain/run1/embed_0.pkl --dest_dir naive --use_meta --attn_type naive
fi

if [ $1 == "selfattn" ] ; then
# self attention model
python misc/combine_meta_process.py --dest_dir /scratch/gjawahar/objects/dywoev/full_v2/selfattn --data_dir /scratch/gjawahar/projects/data/sosweet_lyon/tokenized/tweet_text/zip_stopwords_oui/splits/full_v2/main --feat_dir /scratch/gjawahar/projects/data/sosweet_lyon/tokenized/tweet_text/zip_stopwords_oui/splits/full_v2
CUDA_VISIBLE_DEVICES=2 python premodel/pretrain.py --source_dir /scratch/gjawahar/objects/dywoev/full_v2/selfattn --n_epochs 10000 --n_iter 1 --ns 20 --gpus 1 --neg_dir /scratch/gjawahar/projects/data/sosweet_lyon/tokenized/tweet_text/zip_stopwords_oui/splits/full_v2/intrinsic/ll_neg --dest_dir run1 --skip_eval
CUDA_VISIBLE_DEVICES=2 python metamodel/metatrain.py --source_dir /scratch/gjawahar/objects/dywoev/full_v2/selfattn --neg_dir /scratch/gjawahar/projects/data/sosweet_lyon/tokenized/tweet_text/zip_stopwords_oui/splits/full_v2/intrinsic/ll_neg --n_epochs 10000 --n_iter 9 --ns 20 --gpus 1 --meta_ids user,user,user,user,user,user,user,user,tweet --meta_types embed,embed,category,category,category,embed,presence,category,category --meta_path content/all_tweets_50.txt,content/geo_tweets_50.txt,dept/feat.txt,income/insee_feat.txt,income/iris_feat.txt,network/feat.txt,presence/feat.txt,region/feat.txt,yt_category/feat.txt --init_emb /scratch/gjawahar/objects/dywoev/full_v2/selfattn/result/pretrain/run1/embed_0.pkl --dest_dir self --use_meta --attn_type self --skip_eval
fi

if [ $1 == "contextattn" ] ; then
# context attention model
python misc/combine_meta_process.py --dest_dir /scratch/gjawahar/objects/dywoev/full_v2/contextattn --data_dir /scratch/gjawahar/projects/data/sosweet_lyon/tokenized/tweet_text/zip_stopwords_oui/splits/full_v2/main --feat_dir /scratch/gjawahar/projects/data/sosweet_lyon/tokenized/tweet_text/zip_stopwords_oui/splits/full_v2
CUDA_VISIBLE_DEVICES=3 python premodel/pretrain.py --source_dir /scratch/gjawahar/objects/dywoev/full_v2/contextattn --n_epochs 10000 --n_iter 1 --ns 20 --gpus 1 --neg_dir /scratch/gjawahar/projects/data/sosweet_lyon/tokenized/tweet_text/zip_stopwords_oui/splits/full_v2/intrinsic/ll_neg --dest_dir run1 --skip_eval
CUDA_VISIBLE_DEVICES=3 python metamodel/metatrain.py --source_dir /scratch/gjawahar/objects/dywoev/full_v2/contextattn --neg_dir /scratch/gjawahar/projects/data/sosweet_lyon/tokenized/tweet_text/zip_stopwords_oui/splits/full_v2/intrinsic/ll_neg --n_epochs 10000 --n_iter 9 --ns 20 --gpus 1 --meta_ids user,user,user,user,user,user,user,user,tweet --meta_types embed,embed,category,category,category,embed,presence,category,category --meta_path content/all_tweets_50.txt,content/geo_tweets_50.txt,dept/feat.txt,income/insee_feat.txt,income/iris_feat.txt,network/feat.txt,presence/feat.txt,region/feat.txt,yt_category/feat.txt --init_emb /scratch/gjawahar/objects/dywoev/full_v2/contextattn/result/pretrain/run1/embed_0.pkl --dest_dir context --use_meta --attn_type context --skip_eval
fi

source deactivate

