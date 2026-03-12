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
#SBATCH --array=1,3,7-9,11-15,17,19-28,30,31,34,35 # Specify the range of array jobs (e.g., 0-2 for 3 configs)
# Command(s) to run:
# Define an array of config files
KWARGS_ARRAY=(
    "--model-identifier slurm32260895task0"
    "--model-identifier slurm32260895task1"
    "--model-identifier slurm32260895task2"
    "--model-identifier slurm32260895task3"
    "--model-identifier slurm32273742task4"
    "--model-identifier slurm32260895task5"
    "--model-identifier slurm32260895task6"
    "--model-identifier slurm32260895task7"
    "--model-identifier slurm32260895task8"
    "--model-identifier slurm32260895task9"
    "--model-identifier slurm32260895task10"
    "--model-identifier slurm32260895task11"
    "--model-identifier slurm32260926task12"
    "--model-identifier slurm32260926task13"
    "--model-identifier slurm32263095task14"
    "--model-identifier slurm32263095task15"
    "--model-identifier slurm32263095task16"
    "--model-identifier slurm32263095task17"
    "--model-identifier slurm32263095task18"
    "--model-identifier slurm32263095task19"
    "--model-identifier slurm32263095task20"
    "--model-identifier slurm32263095task21"
    "--model-identifier slurm32263095task22"
    "--model-identifier slurm32263095task23"
    "--model-identifier slurm32263095task24"
    "--model-identifier slurm32263095task25"
    "--model-identifier slurm32263095task26"
    "--model-identifier slurm32263821task27"
    "--model-identifier slurm32263821task28"
    "--model-identifier slurm32263821task29"
    "--model-identifier slurm32265241task30"
    "--model-identifier slurm32265241task31"
    "--model-identifier slurm32292528task32"
    "--model-identifier slurm32292528task33"
    "--model-identifier slurm32292528task34"
    "--model-identifier slurm32293065task35"
    "--model-identifier slurm32260895task0 --true-conditioning-state-weight 1.0"
)

# Get the config file for this array task
KWARGS=${KWARGS_ARRAY[$SLURM_ARRAY_TASK_ID]}

source activate methylseqnet
python ../methylseqnet/predict.py $KWARGS \
    --dataset-keys atlas \
    --dataset-files /global/scratch/projects/vector_streetslab/oberon/methylseqnet_manuscript_datasets/atlas/fold3.h5