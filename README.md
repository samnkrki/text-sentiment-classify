Tips to run via conda
conda create -n sentimentenv python=3.8 -y
conda install -c anaconda seaborn -y
conda install -c conda-forge matplotlib -y
conda install -c conda-forge transformers -y
conda install pytorch::pytorch torchvision torchaudio -c pytorch -y ------> for mac