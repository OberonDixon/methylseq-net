#!/bin/bash
#SBATCH --job-name=train_methylseqnet_factorized_2gpu
#SBATCH --account=fc_nilah
#SBATCH --partition=savio3_gpu
#SBATCH --qos=savio_lowprio
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --requeue
#SBATCH --time=48:00:00
#SBATCH --output=/clusterfs/nilah/oberon/lightning/sbatch_logs/train_methylseqnet_%A_%a.out
#SBATCH --error=/clusterfs/nilah/oberon/lightning/sbatch_logs/train_methylseqnet_%A_%a.err
#SBATCH --exclude=n0134.savio3,n0135.savio3,n0136.savio3,n0137.savio3,n0138.savio3,n0143.savio3,n0144.savio3,n0145.savio3,n0158.savio3,n0159.savio3,n0160.savio3,n0161.savio3,n0174.savio3,n0175.savio3,n0176.savio3
#SBATCH --array=1,3-6 # Specify the range of array jobs (e.g., 0-2 for 3 configs)
# Command(s) to run:
# Define an array of config files
CONFIG_FILES=(
    "../configs/factorized/borzoi_factorized_dual-longread_reps-tm1.0-methyl-multiply-mrep-loss.gin"
    "../configs/factorized/borzoi_factorized_dual-longread+atlas_reps-tm1.0-methyl-multiply-mrep-loss.gin"
    "../configs/factorized/borzoi_factorized_dual-reps-tm1.0-methyl-multiply-mrep-loss.gin"
    "../configs/factorized/borzoi_factorized_dual-longread_reps-tm1.0-methyl-multiply-dilate7x2-no-aux-losses.gin"
    "../configs/factorized/borzoi_factorized_dual-longread+atlas_reps-tm1.0-methyl-multiply-dilate7x2-mrep-loss.gin"
    "../configs/factorized/borzoi_factorized_dual-reps-tm1.0-methyl-multiply-dilate7x2-mrep-loss.gin"
    "../configs/factorized/borzoi_factorized_dual-reps-tm0.0-methyl-multiply-no-aux-losses.gin"
)

# Get the config file for this array task
CONFIG_FILE=${CONFIG_FILES[$SLURM_ARRAY_TASK_ID]}

UNIQUE_IDENTIFIER="slurm${SLURM_ARRAY_JOB_ID}task${SLURM_ARRAY_TASK_ID}"
source activate methylseqnet
python ../methylseqnet/trainer.py --config $CONFIG_FILE --unique_identifier $UNIQUE_IDENTIFIER --batch_size 1
