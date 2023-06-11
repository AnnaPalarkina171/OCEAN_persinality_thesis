#!/bin/bash
#SBATCH --job-name=e-bert            
#SBATCH --output=../outputs/te-bertest-%j.log       
#SBATCH --time=03:00:00                     
#SBATCH --cpus-per-task=2                   

module load Python/PyTorch_GPU_v1.11  

srun python ../e_bert.py  -p Экстраверсия -e 1 --maxl 50