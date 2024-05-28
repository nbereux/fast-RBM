#!/bin/bash

## Mesh files
python scripts/compute_mesh.py -d MICKEY --variable_type Ising --dimension 0 1 --border_length 0.04 --n_pts_dim 100 --device cuda -o ./experiments_paper/mesh_MICKEY.h5 
python scripts/compute_mesh.py -d GENE --variable_type Ising --dimension 0 1 2 --border_length 0.04 --n_pts_dim 100 --device cuda -o ./experiments_paper/mesh_GENE.h5 
python scripts/compute_mesh.py -d MNIST --subset_labels 0 1 --variable_type Ising --dimension 0 1 2 3 --border_length 0.04 --n_pts_dim 50 --device cuda -o ./experiments_paper/mesh_MNIST01.h5 

## Train RCM
python scripts/train_rcm.py -d MICKEY --variable_type Ising --num_hidden 100 --max_iter 100000 --adapt --stop_ll 0.001 --decimation --dimension 0 1 --mesh_file ./experiments_paper/mesh_MICKEY.h5 -o ./experiments_paper/RCM_MICKEY.h5
python scripts/train_rcm.py -d GENE --variable_type Ising --num_hidden 200 --max_iter 100000 --adapt --stop_ll 0.001 --decimation --dimension 0 1 2 --mesh_file ./experiments_paper/mesh_GENE.h5 -o ./experiments_paper/RCM_GENE.h5
python scripts/train_rcm.py -d MNIST --subset_labels 0 1 --variable_type Ising --num_hidden 200 --max_iter 100000 --adapt --stop_ll 0.001 --decimation --dimension 0 1 2 3 --mesh_file ./experiments_paper/mesh_MNIST01.h5 -o ./experiments_paper/RCM_MNIST01.h5

## Convert to RBM
python scripts/rcm_to_rbm.py -i ./experiments_paper/RCM_MICKEY.h5 -o ./experiments_paper/RBM_MICKEY.h5 --num_hiddens 100 -d MICKEY --gibbs_steps 100 --batch_size 2000 --num_chains 2000 --dtype float --device cuda
python scripts/rcm_to_rbm.py -i ./experiments_paper/RCM_GENE.h5 -o ./experiments_paper/RBM_GENE.h5 --num_hiddens 185 -d GENE --gibbs_steps 100 --batch_size 2000 --num_chains 2000 --dtype float --device cuda
python scripts/rcm_to_rbm.py -i ./experiments_paper/RCM_MNIST01.h5 -o ./experiments_paper/RBM_MNIST01.h5 --num_hiddens 200 -d MNIST --subset_labels 0 1 --gibbs_steps 100 --batch_size 2000 --num_chains 2000 --dtype float --device cuda

## Continue training
python scripts/train_rbm.py -d MICKEY --variable_type Bernoulli --use_torch --model BernoulliBernoulliPCDRBM --filename ./experiments_paper/RBM_MICKEY.h5 --epochs 10000 --n_save 1000 --log --dtype float --restore
python scripts/train_rbm.py -d GENE --variable_type Bernoulli --use_torch --model BernoulliBernoulliPCDRBM --filename ./experiments_paper/RBM_GENE.h5 --epochs 10000 --n_save 1000 --log --dtype float --restore
python scripts/train_rbm.py -d MNIST --subset_labels 0 1 --variable_type Bernoulli --use_torch --model BernoulliBernoulliPCDRBM --filename ./experiments_paper/RBM_MNIST01.h5 --epochs 10000 --n_save 1000 --log --dtype float --restore

## Train from scratch
python scripts/train_rbm.py -d MICKEY --variable_type Bernoulli --use_torch --filename ./experiments_paper/JarRBM_MICKEY_from_scratch.h5 --epochs 10000 --n_save 1000 --log --dtype float --model BernoulliBernoulliJarJarRBM --learning_rate 0.01 --num_hiddens 100 --gibbs_steps 100 --batch_size 2000 --num_chains 2000 --min_eps 0.7
python scripts/train_rbm.py -d GENE --variable_type Bernoulli --use_torch --filename ./experiments_paper/JarRBM_GENE_from_scratch.h5 --epochs 10000 --n_save 1000 --log --dtype float --model BernoulliBernoulliJarJarRBM --learning_rate 0.01 --num_hiddens 100 --gibbs_steps 100 --batch_size 2000 --num_chains 2000 --min_eps 0.7
python scripts/train_rbm.py -d MNIST --subset_labels 0 1 --variable_type Bernoulli --use_torch --filename ./experiments_paper/JarRBM_MNIST01_from_scratch.h5 --epochs 10000 --n_save 1000 --log --dtype float --model BernoulliBernoulliJarJarRBM --learning_rate 0.01 --num_hiddens 100 --gibbs_steps 100 --batch_size 2000 --num_chains 2000 --min_eps 0.7
python scripts/train_rbm.py -d MICKEY --variable_type Bernoulli --use_torch --filename ./experiments_paper/RBM_MICKEY_from_scratch.h5 --epochs 10000 --n_save 1000 --log --dtype float --model BernoulliBernoulliPCDRBM --learning_rate 0.01 --num_hiddens 100 --gibbs_steps 100 --batch_size 2000 --num_chains 2000
python scripts/train_rbm.py -d GENE --variable_type Bernoulli --use_torch --filename ./experiments_paper/RBM_GENE_from_scratch.h5 --epochs 10000 --n_save 1000 --log --dtype float --model BernoulliBernoulliPCDRBM --learning_rate 0.01 --num_hiddens 185 --gibbs_steps 100 --batch_size 2000 --num_chains 2000
python scripts/train_rbm.py -d MNIST --subset_labels 0 1 --variable_type Bernoulli --use_torch --filename ./experiments_paper/RBM_MNIST01_from_scratch.h5 --epochs 10000 --n_save 1000 --log --dtype float --model BernoulliBernoulliPCDRBM --learning_rate 0.01 --num_hiddens 200 --gibbs_steps 100 --batch_size 2000 --num_chains 2000 


## PTT sampling 
python scripts/ptt_sampling.py -i ./experiments_paper/RBM_MICKEY.h5 -o sample_RBM_MICKEY.h5 --filename_rcm ./experiments_paper/RCM_MICKEY.h5 --num_samples 10000 --target_acc_rate 0.3 --it_mcmc 1000
python scripts/ptt_sampling.py -i ./experiments_paper/RBM_GENE.h5 -o sample_RBM_GENE.h5 --filename_rcm ./experiments_paper/RCM_GENE.h5 --num_samples 10000 --target_acc_rate 0.3 --it_mcmc 1000
python scripts/ptt_sampling.py -i ./experiments_paper/RBM_MNIST01.h5 -o sample_RBM_MNIST01.h5 --filename_rcm ./experiments_paper/RCM_MNIST01.h5 --num_samples 10000 --target_acc_rate 0.3 --it_mcmc 1000

python scripts/ptt_sampling.py -i ./experiments_paper/RBM_MICKEY_from_scratch.h5 -o sample_RBM_MICKEY_from_scratch.h5 --num_samples 10000 --target_acc_rate 0.3 --it_mcmc 1000
python scripts/ptt_sampling.py -i ./experiments_paper/RBM_GENE_from_scratch.h5 -o sample_RBM_GENE_from_scratch.h5 --num_samples 10000 --target_acc_rate 0.3 --it_mcmc 1000
python scripts/ptt_sampling.py -i ./experiments_paper/RBM_MNIST01_from_scratch.h5 -o sample_RBM_MNIST01_from_scratch.h5 --num_samples 10000 --target_acc_rate 0.3 --it_mcmc 1000

python scripts/ptt_sampling.py -i ./experiments_paper/JarRBM_MICKEY_from_scratch.h5 -o ./experiments_paper/sample_JarRBM_MICKEY_from_scratch.h5 --num_samples 10000 --target_acc_rate 0.3 --it_mcmc 1000
python scripts/ptt_sampling.py -i ./experiments_paper/JarRBM_GENE_from_scratch.h5 -o ./experiments_paper/sample_JarRBM_GENE_from_scratch.h5 --num_samples 10000 --target_acc_rate 0.3 --it_mcmc 1000
python scripts/ptt_sampling.py -i ./experiments_paper/JarRBM_MNIST01_from_scratch.h5 -o ./experiments_paper/sample_JarRBM_MNIST01_from_scratch.h5 --num_samples 10000 --target_acc_rate 0.3 --it_mcmc 1000

