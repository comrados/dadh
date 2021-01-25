#!/bin/bash
#SBATCH -o /home/users/m/mikriukov/projects/dadh/out_nogpu.log
#SBATCH -J dadh
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=TestAndBuild
#SBATCH --time=5:00:00

echo "Loading venv..."
source /home/users/m/mikriukov/venvs/DADH/bin/activate

echo "Executing..."
python3 main.py train --flag ucm --proc nogpu --device cpu