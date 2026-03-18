# README for ALLSETS

## Install dependencies
```
pip install -r requirements.txt

pip install torch_geometric


#Gives some output like 2.10.0, used in the next install
python -c "import torch; print(torch.__version__)"


# Use output from last command here. Use CPU, unless you want and have a GPU, then run next commands
pip install torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-${torch.version}+cpu.html

#Only commands if GPU, not tested since I have no GPU. This gives some output like 12.8
python -c "import torch; print(torch.version.cuda)"

#Now use output from last command
pip install torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-${torch.version}+${CUDA.version}.html


#Run this from the root
pip install -e .
```

