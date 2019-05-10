#!/bin/bash
#
# At first let's give job some descriptive name to distinct
# from other jobs running on cluster
#SBATCH -J Aligner
#
# Let's redirect job's out some other file than default slurm-%jobid-out
#SBATCH --output=log/log.txt
#SBATCH --error=log/log.txt
#
# We'll want to allocate one CPU core
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#
# We'll want to reserve 2GB memory for the job
# and 3 days of compute time to finish. 
# Also define to use the GPU partition.
#SBATCH --mem=25000
#SBATCH --time=6-23:59
#SBATCH --partition=normal
#
# These commands will be executed on the compute node:

# Load all modules you need:

source /home/opt/anaconda3/bin/activate tf

python train_shape_predictors.py

