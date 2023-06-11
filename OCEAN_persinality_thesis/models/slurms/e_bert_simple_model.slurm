#!/bin/bash
#SBATCH --job-name=e-bert            
#SBATCH --output=../outputs/e-bert_simple_model-%j.log       
#SBATCH --time=05:00:00                     
#SBATCH --cpus-per-task=2                   
#SBATCH --gpus=1 

module load Python/PyTorch_GPU_v1.11  

srun python ../e_bert_simple_model.py  
