#!/bin/bash
#SBATCH --job-name=seg41
#SBATCH --time=18:00:00 ## 18 hours for our data
#SBATCH --nodes 1  
#SBATCH --ntasks-per-node=1
#SBATCH --mem 75G
#SBATCH -c 10
#SBATCH -p gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --error=./cyprus/errors/error_nc.o%j
#SBATCH --output=./cyprus/outputs/output_nc.o%j

## if you want your job to run only after another job has run, remove one # and enter that job's ID here (which you can get by running squeue -u your_username)
##SBATCH --dependency=afterok:(job ID) ## without the parentheses, ex afterok:123456

## add your email here if you want to be notified of any changes
#SBATCH --mail-user=
#SBATCH --mail-type=ALL

## These are just general commands
echo "running in shell: " "$SHELL"
export NCCL_SOCKET_IFNAME=lo

## load miniconda, activate your environment, and set the TMPDIR to your folder on the shared drive in case your program creates any temporary files
## load miniconda, activate your environment, and set the TMPDIR to your folder on the shared drive in case your program creates any temporary files
spack load miniconda3
eval "$(conda shell.bash hook)"
conda activate /minconda3/envs/sam_cloned_env # path to environment
export TMPDIR='' # path to your folder

## run your program
## change to actual path of the program and environment 
## session refers to the folder where the data is stored
srun /minconda3/envs/sam_cloned_env/bin/python /src/runner_get_seg.py --session Expl_ --start_frame 8600 --end_frame 27820
