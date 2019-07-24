conda create -n dynword_pytorch36 python=3.6 --yes
source activate dynword_pytorch36
conda install pytorch torchvision -c pytorch --yes
pip install tqdm
pip install cython
pip install gensim==3.4.0
pip install pandas==0.23.0
pip install scikit-learn==0.19.1s
pip install spherecluster