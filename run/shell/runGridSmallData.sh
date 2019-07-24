#!/bin/sh
# This script runs the dynamic word embedding models for the partial corpus for different hyper-parameters
# oarsub -l /nodes=5/cpu=2/core=4,walltime=150:00:00 ./runGridSmallData.sh

# set the following
#DATA_FOLDER=/home/ganesh/data/sosweet/full
#OBJ_FOLDER=/home/ganesh/objects/dywoev/sept28_meta_search
#CODE_FOLDER=/home/ganesh/projects/tmp/dywoev/sept28_meta_search
DATA_FOLDER=/home/alpaga/ganesh/data/sosweet/full
OBJ_FOLDER=/home/alpaga/ganesh/objects/dywoev/sept30_meta_search
CODE_FOLDER=/home/alpaga/ganesh/projects/dywoev
#DATA_FOLDER=/scratch/gjawahar/projects/data/sosweet_lyon/tokenized/tweet_text/zip_stopwords_oui/splits/full
#OBJ_FOLDER=/scratch/gjawahar/projects/objects/dywoev/grid_aug2
#CODE_FOLDER=/scratch/gjawahar/projects/dywoev

rm -rf $OBJ_FOLDER
mkdir $OBJ_FOLDER

source activate dynword_pytorch36

#python $CODE_FOLDER/run/py/grid_search.py $DATA_FOLDER $OBJ_FOLDER $CODE_FOLDER $OBJ_FOLDER/tmp
python $CODE_FOLDER/run/py/meta_search.py $DATA_FOLDER $OBJ_FOLDER $CODE_FOLDER $OBJ_FOLDER/tmp

# create data
dataprocess(){
  source activate dynword_pytorch36
  eval $1
  source deactivate
  exit 0
}
export -f dataprocess
commands=()
while read line
do
  commands+=("$line")
done < $OBJ_FOLDER/tmp.data
SECONDS=0
for command in "${commands[@]}";
do
  echo $command
done  | xargs -I {} --max-procs 9 bash -c 'dataprocess "$@"' _ {}
duration=$SECONDS
echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds took to create data." >> /tmp/dywoev/out_sept30_meta_search

# run pretraining
trainprocess(){
  source activate dynword_pytorch36
  eval $1
  source deactivate
  exit 0
}
export -f trainprocess
commands=()
while read line
do
  commands+=("$line")
done < $OBJ_FOLDER/tmp.premodel
SECONDS=0
for command in "${commands[@]}";
do
  echo $command
done  | xargs -I {} --max-procs 4 bash -c 'trainprocess "$@"' _ {}
duration=$SECONDS
echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds took to pretrain." >> /tmp/dywoev/out_sept30_meta_search

# run dyntraining
commands=()
while read line
do
  commands+=("$line")
done < $OBJ_FOLDER/tmp.dynmodel
SECONDS=0
for command in "${commands[@]}";
do
  echo $command
done  | xargs -I {} --max-procs 4 bash -c 'trainprocess "$@"' _ {}
duration=$SECONDS
echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds took to dyntrain." >> /tmp/dywoev/out_sept30_meta_search

# cleanup
bash $OBJ_FOLDER/tmp.post

source deactivate


