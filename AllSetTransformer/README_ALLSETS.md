# Digestible README for ALLSETS

Ran it in bash, using miniconda3 ([Miniconda3 Link](https://www.anaconda.com/docs/getting-started/miniconda/install#windows-powershell))
## Setting up environment (This should be changed to venv in the future)
Currently miniconda is used, since that is what Allset originally did, this should however be changed to venv, since that is what we use for the other implementation

```
source /c/Users/{user_name}/miniconda3/etc/profile.d/conda.sh
conda activate AllSet
```

## Install dependencies
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.7.0+cu118.html
pip install ipdb tqdm scipy matplotlib
```

## Folder creation for data
Create 2 folders:
```
data/dgl_data_raw
data/pyg_data/hypergraph_dataset_updated
```
Unzip AllSet_all_raw_data.zip into data

This should create a folder data/AllSet_all_raw_data
## Running the program (Allset implementation)
Code still no run since they use old methods or methods in old ways, so now time to fix each issue one at a time
```
bash run_AllSetTransformer.sh
```
in src directory

## Running the program (Our implenetaion &ndash; subject to change)
We don't have any shell scripts to run the code and in our code the parameters are hardcoded in, instead of running a Shell script, therefore you should use the following command in the AllSetTransformer directory:

```
python train.py
```


Allset does 20 runs and 500 epochs for the Yelp data set, this can be configured in *train.py*