#!/bin/bash

##======= Train prior =======##

# Train prior on Chembl_32 HELM sequences to learn HELM grammar
# python train_prior.py --train_data data/prior/chembl32/biotherapeutics_dict_prot_flt.csv --valid_data data/prior/chembl32/biotherapeutics_dict_prot_flt.csv --output_dir result/prior/chembl_5.0 --n_epochs 200 --max_len 200 --batch_size 1024

# Sample 1000 sequences from the prior for evaluation
# python generate.py --model_path result/prior/chembl_5.0/gpt_model_34_0.143.pt --out_file result/prior/chembl_5.0/1k_samples.csv --n_samples 1000 --max_len 200 --batch_size 128

##======= Cyc CPP design =======##
## Directly fine-tune on CycPeptMDB dataset
# python train_prior.py --model_path result/prior/chembl_5.0/gpt_model_34_0.143.pt \
#  --train_data data/prior/CycPeptMPDB/CycPeptMPDB_Peptide_All_flt.csv --valid_data data/prior/CycPeptMPDB/CycPeptMPDB_Peptide_All_flt.csv --output_dir result/prior/cycpeptpdb_tune_5.0 --n_epochs 200 --max_len 200 --batch_size 1024
# python generate.py --model_path result/prior/cycpeptpdb_tune_5.0/gpt_model_20_0.075.pt --out_file result/prior/cycpeptpdb_tune_5.0/1k_samples.csv --n_samples 1000 --max_len 200 --batch_size 128

## Finetune on the top 1000 sequences with the highest permeability
# python train_prior.py --model_path result/prior/cycpeptpdb_tune_5.0/gpt_model_20_0.075.pt \
#  --train_data data/prior/prior_perm_top1k.csv --valid_data data/prior/prior_perm_top1k.csv --output_dir result/prior/perm_tune --n_epochs 10 --max_len 200 --batch_size 64
# python generate.py --model_path result/prior/perm_tune/gpt_model_final_0.076.pt  --out_file result/prior/perm_tune/1k_samples.csv --n_samples 1000 --max_len 200 --batch_size 128

## Train agent for CPP ##
# python train_agent.py --prior result/prior/perm_tune/gpt_model_final_0.076.pt --output_dir result/agent/cpp/pep_perm_5.1_reinvent --batch_size 32 --n_steps 500 --sigma 60 --task permeability  --max_len 140
# python train_agent.py --prior result/prior/perm_tune/gpt_model_final_0.076.pt --output_dir result/agent/cpp/pep_perm_5.1_reinvent_cpl_a1 --batch_size 32 --n_steps 500 --sigma 60 --task permeability  --max_len 140 --loss_type reinvent_cpl --alpha 1.0
# python generate.py --model_path result/agent/cpp/pep_perm_5.1_reinvent_cpl_a1_bs100_3000s/Agent_final_0.679.pt  --out_file result/agent/cpp/pep_perm_5.1_reinvent_cpl_a1_bs100_3000s/1k_samples.csv --n_samples 1000 --max_len 200 --batch_size 128

##======= KRas binder design =======##
## Directly fine-tune on KRas Kd and CycPetMPDB dataset
# python train_prior.py --model_path result/prior/chembl_5.0/gpt_model_34_0.143.pt \
#  --train_data data/kras_kd/kras_kd_prior.csv --valid_data data/kras_kd/kras_kd_prior.csv --output_dir result/prior/kras_kd_tune_5.0 --n_epochs 200 --max_len 200 --batch_size 256
# python generate.py --model_path result/prior/kras_kd_tune_5.0/gpt_model_7_0.074.pt  --out_file result/prior/kras_kd_tune_5.0/1k_samples.csv --n_samples 1000 --max_len 200 --batch_size 128

## Finetune on the top 1000 sequences with the lowest kd
# python train_prior.py --model_path result/prior/kras_kd_tune_5.0/gpt_model_final_0.322.pt \
#     --train_data data/prior/prior_kras_kd_top1k.csv --valid_data data/prior/prior_kras_kd_top1k.csv --output_dir result/prior/kras_kd_tune --n_epochs 10 --max_len 200 --batch_size 64
# python generate.py --model_path result/prior/kras_kd_tune/gpt_model_final_0.096.pt  --out_file result/prior/kras_kd_tune/1k_samples.csv --n_samples 1000 --max_len 200 --batch_size 128

# python train_agent.py --prior result/prior/kras_kd_tune/gpt_model_final_0.096.pt --output_dir result/agent/kras_kd/kras_5.1 --batch_size 32 --n_steps 500 --sigma 60 --task kras_kd  --max_len 140
# python generate.py --model_path result/agent/kras_kd/kras_5.1_reinvent_cpl-a1_bs100_3000s_v2/Agent_final_0.678.pt  --out_file result/agent/kras_kd/kras_5.1_reinvent_cpl-a1_bs100_3000s_v2/1k_samples.csv --n_samples 1000 --max_len 200 --batch_size 128


##======= KRAS CPP opt =======##
# python train_agent.py --prior result/prior/kras_perm_tune/gpt_model_final_0.088.pt --output_dir result/agent/kras_perm/kras_perm_5.4_reinvent_cpl_test  --batch_size 32 --n_steps 500 --sigma 60 --task kras_perm  --max_len 140 --alpha 1.0 --loss_type reinvent_cpl

##======= KRAS CPP opt -- step 2 =======##
# python train_prior.py --model_path result/prior/cycpeptpdb_tune_5.0/gpt_model_20_0.075.pt \
#  --train_data result/agent/cpp/pep_perm_5.1_reinvent_cpl_a1_bs100_3000s/both_opt_prior.csv --valid_data result/agent/cpp/pep_perm_5.1_reinvent_cpl_a1_bs100_3000s/both_opt_prior.csv --output_dir result/prior/perm_kras_tune_5.1 --n_epochs 20 --max_len 200 --batch_size 256
# python train_agent.py --prior result/prior/perm_kras_tune_5.1/gpt_model_final_0.105.pt --output_dir result/agent/kras_perm/kras_perm_5.5_reinvent_cpl_test  --batch_size 32 --n_steps 500 --sigma 60 --task kras_perm  --max_len 140 --alpha 1.0 --loss_type reinvent_cpl