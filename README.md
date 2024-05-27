# fast-RBM
Code for NeurIPS 2024 submission #2003


## Installation

/!\ To install PyTorch with GPU support, see [this link](https://pytorch.org/).

To install the package, you need to
```bash
git clone https://github.com/nbereux/fast-RBM.git
cd fast-RBM
pip install -r requirements.txt
pip install -e .
```

## Usage 
All scripts can be called with `--help` options to get a description of the arguments

### 1. Mesh  
Compute the mesh on the intrinsic space 
 ```bash
python scripts/compute_mesh.py -d MICKEY --variable_type Ising --dimension 0 1 --border_length 0.04 --n_pts_dim 100 --device cuda -o ./mesh.h5 
 ```
#### Arguments 
  - `-d` is the name of the dataset to load ("MICKEY", "MNIST" or "GENE")
  - `--variable_type` should be set to Ising for the mesh. Can be ("Ising", "Bernoulli", "Continuous" or "Potts"). Currently, only "Ising" works for the RCM.
  - `--dimension` is the index of the dimensions of the intrinsic space
  - `--border_length` should be set as 2/50 or less
  - `--n_pts_dim` is the number of points of the mesh for each dimension. The total number of points will be `n_pts_dim**n_dim`
  - `--device` is the pytorch device you want to use. On lower dimensions, the CPU and GPU have similar performance.
  - `-o` The filename for your mesh.

### 2. RCM
Train the RCM and compute the corresponding Ising RBM:
```bash
python scripts/train_rcm.py -d MICKEY --variable_type Ising --num_hidden 200 --max_iter 100000 --adapt --stop_ll 0.001 --decimation --dimension 0 1 --mesh_file mesh.h5 -o RCM.h5
```
#### Arguments 
 - `--num_hidden` is the maximum number of hidden nodes for the final RBM.
 - `--max_iter` is the maximum training iterations allowed before stopping.
 - `--adapt` allows to use an adaptive learning rate strategy.
 - `--stop_ll` is the threshold for the exponential moving average on the test log likelihood fluctuations.
 - `--decimation` allows for removal of unimportant features to improve the mapping from RCM to Ising RBM. If not set the final RBM will have exactly `--num_hidden` hidden nodes.
 - `--mesh_file` path to the mesh computed at step 1.

### 3. Convert the Ising-RBM to Bernoulli-RBM
```bash
python scripts/rcm_to_rbm.py -i RCM.h5 -o RBM.h5 --num_hiddens 100 -d MICKEY --gibbs_steps 100 --batch_size 2000 --num_chains 2000 --min_eps 0.7 --dtype float --device cuda
```
#### Arguments 
 - `-i` is the filename for the RCM obtained at step 2
 - `-o` is the filename for the new RBM initialized with the RCM.
 - `--num_hiddens` The target number of hidden nodes for the RBM. If below the final number of hidden nodes of the RCM will do nothing. Otherwise the new nodes are initialized with 0 bias and random weights.
 - `--gibbs_steps` The number of gibbs steps performed at each gradient updates
 - `--num_chains` The number of parallel chains used for the gradient estimation. 
 - `--dtype` the dtype for the weights of the RBM.
 - `--min_eps` Minimum effective population size for the Jar-RBM.

### 4. Continue the training
```bash
python scripts/train_rbm.py -d MICKEY --variable_type Bernoulli --use_torch --model BernoulliBernoulliJarJarRBM --filename RBM.h5 --epochs 1000 --log --dtype float --restore
```
#### Arguments  
 - `--use_torch` loads the dataset entirely on the GPU allowing for faster processing in exchange for higher VRAM footprint
 - `--model` The algorithm to train the RBM. Can be `BernoulliBernoulliJarJarRBM` or `BernoulliBernoulliPCDRBM`
 - `--filename` path to the RBM you want to continue training.
 - `--epochs` total number of training epochs for the model.
 - `--log` write metrics in a log file.
 - `--restore` Must be put to restore training. Otherwise will start a new training.

 ### 5. Train from scratch
 ```bash
python scripts/train_rbm.py -d MICKEY --variable_type Bernoulli --use_torch --filename RBM.h5 --epochs 1000 --log --dtype float --model BernoulliBernoulliJarJarRBM --learning_rate 0.01 --num_hiddens 100 --gibbs_steps 100 --batch_size 2000 --num_chains 2000 --min_eps 0.7
 ```

### 6. PTT
```bash
python scripts/ptt_sampling.py -i RBM.h5 -o sample_RBM_mickey.h5 --num_samples 2000 --target_acc_rate 0.9 --it_mcmc 1000
```
#### Arguments 
- `-i` is the filename of the RBM obtained at step 4 or 5.
- `-o` is the file in which to save the samples.
- `--filename_rcm` the name of the file used to initialize the RBM at step 3. Do not set if the RBM was trained from scratch
- `--target_acc_rate` The target acceptance rate between two consecutive machines
- `--it_mcmc` The number of gibbs steps performed by each machine.

## Analysis
See [rcm_analysis.ipynb](notebooks/rcm_analysis.ipynb) for an analysis of the file obtained at step 2 and [rbm_analysis.ipynb](notebooks/rbm_analysis.ipynb) for an analysis of the files obtained at step 4 or 5, as well as the results for the PTT.