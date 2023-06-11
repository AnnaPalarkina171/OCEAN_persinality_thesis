#!/bin/bash
#SBATCH --job-name=rnn_3_attention_stacked_over            
#SBATCH --output=../outputs/rnn_3_attention_stacked_over-%j.log       
#SBATCH --time=05:00:00                     
#SBATCH --cpus-per-task=2                   
#SBATCH --gpus=1 

module load Python/PyTorch_GPU_v1.11  

srun python ../rnn_3_attention_stacked.py  -p Нейротизм