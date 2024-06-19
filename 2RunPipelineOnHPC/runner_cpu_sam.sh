#!/bin/bash
#SBATCH --job-name=sam52
#SBATCH --time=18:00:00  # 18 hours for our data
#SBATCH --nodes 1  
#SBATCH --ntasks-per-node=1
#SBATCH --mem 75G
#SBATCH -c 10
#SBATCH -p workq
#SBATCH --error=./cyprus/errors/error_nc.o%j
#SBATCH --output=./cyprus/outputs/output_nc.o%j
## set the email address where you want to be notified of any changes
#SBATCH --mail-user=
#SBATCH --mail-type=ALL

## These are just general commands
echo "running in shell: " "$SHELL"
export NCCL_SOCKET_IFNAME=lo

## load miniconda, activate your environment, and set the TMPDIR to your folder on the shared drive in case your program creates any temporary files
spack load miniconda3
eval "$(conda shell.bash hook)"
conda activate /minconda3/envs/sam_cloned_env ## path to environment
export TMPDIR='' ## path to your folder

## run your program
## change to actual path of the program and environment
srun /minconda3/envs/sam_cloned_env/bin/python /src/runner_pred_sam.py --session Expl --start_frame 7640 --end_frame 26950 

