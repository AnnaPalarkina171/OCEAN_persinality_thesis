#!/bin/bash
#SBATCH --job-name=n-bert            
#SBATCH --output=../outputs/rnn_3-%j.log       
#SBATCH --time=05:00:00                     
#SBATCH --cpus-per-task=2                   
#SBATCH --gpus=1 

module load Python/PyTorch_GPU_v1.11  

srun python ../rnn_3.py  -p Нейротизм