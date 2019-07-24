## Contextualized Diachronic Word Representations
Code used in our [Workshop on Historical Language Change at ACL 2019 paper](https://drive.google.com/open?id=17SrReRC9guEa4ppwg0wwb13AciLWK3Mv).

### Dependencies
* [PyTorch](https://pytorch.org/)
* [Gensim](https://radimrehurek.com/gensim/) (for baselines)
* [pandas](https://pandas.pydata.org/)
* [scikit-learn](https://scikit-learn.org/)

### Quick Start
* Download the data (worth ~31GB) from the following Google Drive link:
```
https://drive.google.com/drive/folders/14QyPrfa-tCsIw2ZtchyZzrJjJlbUQ3ms?usp=sharing
```
and set the environment variable `$DATA_DIR` to point to the extracted folder.
* Train the Word2vec (vanilla) baseline:
```
python word2vec.py --window 2 --logic 0 --workers 8 --out /tmp/vanilla
```
* Train the HistWords baseline:
```
python word2vec.py --window 2 --logic 2 --workers 8 --out /tmp/vanilla
```
* Train the DBE baseline:
```
python misc/preprocess.py --dest_dir /tmp/dbe --src_dir $DATA_DIR/main
python premodel/pretrain.py --source_dir /tmp/dbe --n_epochs 10000 --n_iter 1 --ns 20 --gpus 1 --dest_dir run1 --skip_eval
python dynmodel/dyntrain.py --source_dir /tmp/dbe --init_emb /tmp/dbe/result/pretrain/run1/embed_0.pkl --n_iter 9 --ns 20 --n_epochs 10000 --gpus 1 --dest_dir dbe --skip_eval
python dynmodel/dyneval.py --source_dir /tmp/dbe --ns 20 --n_epochs 10000 --gpus 1 --dest_dir dbe --init_emb /tmp/dbe/result/pretrain/run1/embed_0.pkl
```
* Train the context attention model (set `--attn_type self` for self-attention variant or set `--attn_type naive` for no-attention variant of our proposed model):
```
python misc/combine_meta_process.py --dest_dir /tmp/contextattn --data_dir $DATA_DIR/main --feat_dir $DATA_DIR
python premodel/pretrain.py --source_dir /tmp/contextattn --n_epochs 10000 --n_iter 1 --ns 20 --gpus 1 --neg_dir $DATA_DIR/intrinsic/ll_neg --dest_dir run1 --skip_eval
python metamodel/metatrain.py --source_dir /tmp/contextattn --neg_dir $DATA_DIR/intrinsic/ll_neg --n_epochs 10000 --n_iter 9 --ns 20 --gpus 1 --meta_ids user,user,user,user,user,user,user,user,tweet --meta_types embed,embed,category,category,category,embed,presence,category,category --meta_path content/all_tweets_50.txt,content/geo_tweets_50.txt,dept/feat.txt,income/insee_feat.txt,income/iris_feat.txt,network/feat.txt,presence/feat.txt,region/feat.txt,yt_category/feat.txt --init_emb /tmp/contextattn/result/pretrain/run1/embed_0.pkl --dest_dir context --use_meta --attn_type context --skip_eval
```

### Acknowledgements
This repository would not be possible without the efforts of the creators/maintainers of the following libraries:

### License
This repository is GPL-licensed.


