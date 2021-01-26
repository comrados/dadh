#!/bin/bash
#SBATCH -o /home/users/m/mikriukov/projects/dadh/out_tab.log
#SBATCH -J dadh
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=TestAndBuild
#SBATCH --time=5:00:00
#SBATCH --gres=gpu:tesla:1

echo "Loading venv..."
source /home/users/m/mikriukov/venvs/DADH/bin/activate

echo "Loading cuda..."
module load nvidia/cuda/10.1

echo "Executing..."
python3 main.py train --flag ucm --proc tab