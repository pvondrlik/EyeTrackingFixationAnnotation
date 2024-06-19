#!/bin/bash
#SBATCH --job-name=sam23-25
#SBATCH --time=2:00
#SBATCH --nodes 1  
#SBATCH --ntasks-per-node=1
#SBATCH --mem 20G
#SBATCH -c 1
#SBATCH -p workq

## change path accordingly
#SBATCH --error=./error_nc.o%j
#SBATCH --output=./output_nc.o%j
## add your mail to get notifications
#SBATCH --mail-user=
#SBATCH --mail-type=ALL

## These are just general commands
echo "running in shell: " "$SHELL"
export NCCL_SOCKET_IFNAME=lo

## load miniconda, activate your environment, and set the TMPDIR to your folder on the shared drive in case your program creates any temporary files
spack load miniconda3
eval "$(conda shell.bash hook)"
conda activate /minconda3/envs/sam_cloned_env # Path to environments
export TMPDIR='' # Path to store temporary files

## run your program
srun /minconda3/envs/sam_cloned_env/bin/python /sumlabel.py # change path to your python script and environment

