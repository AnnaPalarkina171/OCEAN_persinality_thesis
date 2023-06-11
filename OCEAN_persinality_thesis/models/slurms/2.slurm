#!/bin/bash
#SBATCH --job-name=3_class            # Название задачи   
#SBATCH --output=../outputs/3_class_bert-%j.log       # Файл для вывода результатов 
#SBATCH --time=04:00:00                      # Максимальное время выполнения 
#SBATCH --cpus-per-task=2                   # Количество CPU на одну задачу 
#SBATCH --gpus=2

module load Python/PyTorch_GPU_v1.11  

nvidia-smi
srun python3 ../3_classes_bert.py -p Экстраверсия --lr 2e-5 -e 8 -ws 10000