#!/bin/bash
#SBATCH --job-name=predict_methylseqnet_atlas
#SBATCH --account=fc_nilah
#SBATCH --partition=savio3_gpu
#SBATCH --qos=savio_lowprio
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --time=6:00:00
#SBATCH --output=/clusterfs/nilah/oberon/lightning/sbatch_logs/predict_methylseqnet_%A_%a.out
#SBATCH --error=/clusterfs/nilah/oberon/lightning/sbatch_logs/predict_methylseqnet_%A_%a.err
#SBATCH --exclude=n0005.savio3,n0217.savio3,n0215.savio3,n0130.savio4,n0132.savio4,n0134.savio3,n0135.savio3,n0136.savio3,n0137.savio3,n0138.savio3,n0143.savio3,n0144.savio3,n0145.savio3,n0158.savio3,n0159.savio3,n0160.savio3,n0161.savio3,n0174.savio3,n0175.savio3,n0176.savio3
#SBATCH --array=0-2 # Specify the range of array jobs (e.g., 0-2 for 3 configs)
# Command(s) to run:
# Define an array of config files
KWARGS_ARRAY=(
    "--model-identifier slurm31983102task0 --true-conditioning-state-weight 1.0"
    "--model-identifier slurm31983102task0 --true-conditioning-state-weight 0.0"
    "--model-identifier slurm31986468task1"
)

# Get the config file for this array task
KWARGS=${KWARGS_ARRAY[$SLURM_ARRAY_TASK_ID]}

source activate methylseqnet
# python ../methylseqnet/predict.py $KWARGS \
#     --dataset-keys longread \
#     --dataset-files /global/scratch/projects/vector_streetslab/oberon/methylseqnet_manuscript_datasets/gm12878/fold3.h5

python ../methylseqnet/predict.py $KWARGS \
    --dataset-keys longread \
    --dataset-files /global/scratch/projects/vector_streetslab/oberon/methylseqnet_manuscript_datasets/gm12878_haplotyped/fold3.h5