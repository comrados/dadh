#!/bin/bash
#SBATCH -o /home/users/m/mikriukov/projects/dadh/out.log
#SBATCH -J dadh
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

echo "Loading venv..."
source /home/users/m/mikriukov/venvs/DADH/bin/activate

echo "Executing..."
python3 main.py