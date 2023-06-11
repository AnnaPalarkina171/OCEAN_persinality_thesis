#!/bin/bash
#SBATCH --job-name=test_myenv            # Название задачи   
#SBATCH --output=../outputs/test_myenv-%j.log       # Файл для вывода результатов 
#SBATCH --time=00:03:00                      # Максимальное время выполнения 
#SBATCH --cpus-per-task=1                   # Количество CPU на одну задачу 

module purge
module load Python

source deactivate
source activate my_py_env


srun python ../test.py  