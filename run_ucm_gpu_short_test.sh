#!/bin/bash
#SBATCH -o /home/users/m/mikriukov/projects/dadh/out_gpu_short.log
#SBATCH -J dadh
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpu_short
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:tesla:1

echo "Loading venv..."
source /home/users/m/mikriukov/venvs/DADH/bin/activate

echo "Loading cuda..."
module load nvidia/cuda/10.1

echo "Executing..."
python3 main.py test --flag ucm --proc short