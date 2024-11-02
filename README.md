# Distributionally Robust Self-supervised Learning for Tabular Data

This repository hosts the code to replicate experiments of the paper "Distributionally Robust Self-supervised Learning for Tabular Data" with FT Transformer backbone.

## Installation
The experiments are performed on the "bank" and "census" datasets from UCI. Use the following links to download the datasets:

https://archive.ics.uci.edu/dataset/222/bank+marketing

https://archive.ics.uci.edu/dataset/20/census+income

After downloading the data, use ```pip``` to install the required packages for this project:

```pip install -r requirements.txt```

## Usage

### JTT on bank dataset
```bash
CUDA_LAUNCH_BLOCKING=1 python experiments_tab_transformer_JTT.py \
--full_csv="bank-additional-full.csv" \
--model_type="FTTransformer" \
--categories job marital education default housing loan contact month day_of_week poutcome \
--num_cols age duration \
--seed=43 \
--max_epoch_phase1A=35 \
--max_epoch_phase1B=100 \
--max_epoch_phase1B=100 \
--batch_size=1024 \
--output_col="y" \
--dim_out=192 \
--mask_val=0.05 \
--upweight_factor=50 \
--dataset="bank" > ft_transformer-mr-0.05-upwt-50.out
```

### JTT on census dataset
```bash
CUDA_LAUNCH_BLOCKING=1 python experiments_tab_transformer_JTT.py \
--full_csv="adult" \
--model_type="FTTransformer" \
--categories workclass education marital-status occupation relationship race sex native-country \
--num_cols age education-num \
--seed=43 \
--max_epoch_phase1A=100 \
--max_epoch_phase1B=200 \
--max_epoch_phase1B=200 \
--batch_size=1024 \
--output_col="income(>=50k)" \
--dim_out=192 \
--mask_val=0.05 \
--upweight_factor=50 \
--dataset="census" > ft_transformer-mr-0.05-upwt-50.out
```

### DFR on bank dataset
```bash
CUDA_LAUNCH_BLOCKING=1 python experiments_tab_transformer_DFR.py \
--full_csv="bank-additional-full.csv" \
--model_type="FTTransformer" \
--categories job marital education default housing loan contact month day_of_week poutcome \
--num_cols age duration \
--seed=43 \
--max_epoch_phase1A=35 \
--max_epoch_phase1B=100 \
--max_epoch_phase2B=100 \
--batch_size=1024 \
--output_col="y" \
--dim_out=192 \
--mask_val=0.05 \
--upweight_factor=50 \
--dataset="bank" > ft_transformer-mr-0.05-upwt-50.out
```

### DFR on census dataset
```bash
CUDA_LAUNCH_BLOCKING=1 python experiments_tab_transformer_DFR.py \
--full_csv="adult" \
--model_type="FTTransformer" \
--categories workclass education marital-status occupation relationship race sex native-country \
--num_cols age education-num \
--seed=43 \
--max_epoch_phase1A=100 \
--max_epoch_phase1B=200 \
--max_epoch_phase2B=200 \
--batch_size=1024 \
--output_col="income(>=50k)" \
--dim_out=192 \
--mask_val=0.05 \
--upweight_factor=50 \
 --dataset="census" > ft_transformer-mr-0.05-upwt-50.out
```

## Acknowledgement
Tab-transformer, and ft-transformer backbones were adapted from https://github.com/lucidrains/tab-transformer-pytorch

## Citation
```
@Inproceedings{Ghosh2024,
 author = {Shantanu Ghosh and Joseph Xie and Mikhail Kuznetsov},
 title = {Distributionally robust self-supervised learning for tabular data},
 year = {2024},
 url = {https://www.amazon.science/publications/distributionally-robust-self-supervised-learning-for-tabular-data},
 booktitle = {NeurIPS 2024 Workshop on Table Representation Learning},
}
```