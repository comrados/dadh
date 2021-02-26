#!/bin/bash
#SBATCH -o /home/users/m/mikriukov/projects/dadh/out_gpu_short.log
#SBATCH -J dadh
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpu_short
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:tesla:1

echo "Loading venv..."
source /home/users/m/mikriukov/venvs/DADH/bin/activate

echo "Loading cuda..."
module load nvidia/cuda/10.1

echo "Training 128..."
python3 main.py train --flag $1 --proc short --bit 128

echo "Testing 128..."
python3 main.py test --flag $1 --proc short --bit 128

echo "Training 64..."
python3 main.py train --flag $1 --proc short --bit 64

echo "Testing 64..."
python3 main.py test --flag $1 --proc short --bit 64

echo "Training 32..."
python3 main.py train --flag $1 --proc short --bit 32

echo "Testing 32..."
python3 main.py test --flag $1 --proc short --bit 32

echo "Training 16..."
python3 main.py train --flag $1 --proc short --bit 16

echo "Testing 16..."
python3 main.py test --flag $1 --proc short --bit 16

echo "Training 8..."
python3 main.py train --flag $1 --proc short --bit 8

echo "Testing 8..."
python3 main.py test --flag $1 --proc short --bit 8