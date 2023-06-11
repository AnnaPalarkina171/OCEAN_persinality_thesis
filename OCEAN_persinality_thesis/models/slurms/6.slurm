#!/bin/bash
#SBATCH --job-name=3_class            # Название задачи   
#SBATCH --output=../outputs/3_class_bert-%j.log       # Файл для вывода результатов 
#SBATCH --time=04:00:00                      # Максимальное время выполнения 
#SBATCH --cpus-per-task=1                   # Количество CPU на одну задачу 
#SBATCH --gpus=1 

module load Python/PyTorch_GPU_v1.11  

srun python ../3_classes_bert.py -p "Открытость опыту"