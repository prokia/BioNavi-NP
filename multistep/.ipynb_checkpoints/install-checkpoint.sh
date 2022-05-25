conda install -y rdkit -c rdkit
conda install -y pytorch torchvision torchaudio  cudatoolkit=11.1 -c pytorch-lts -c nvidia
pip install networkx graphviz tqdm torchtext==0.6.0 configargparse
pip install -e retro_star/packages/mlp_retrosyn
pip install -e retro_star/packages/rdchiral
pip install -e onmt/