#!/bin/bash
#SBATCH --job-name=folds_through_methylseq_1gpu
#SBATCH --account=fc_nilah
#SBATCH --partition=savio4_gpu
#SBATCH --qos=savio_lowprio
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --time=24:00:00
#SBATCH --output=/clusterfs/nilah/oberon/lightning/sbatch_logs/predict_methylseqnet_%A_%a.out
#SBATCH --error=/clusterfs/nilah/oberon/lightning/sbatch_logs/predict_methylseqnet_%A_%a.err
#SBATCH --array=0-3 # Specify the range of array jobs (e.g., 0-2 for 3 configs)

# Command(s) to run:
# Define an array of model paths
#    "slurm30018035task9"
#    "slurm30039609task10"
#    "slurm30024664task1"
#    "slurm29867045task1"
#    "slurm29952986task5"
#    "slurm29940167task0"
#    "slurm29865703task0"
#    "slurm29867045task4"
MODEL_IDENTIFIERS=(
    "slurm30634711task0"
    "slurm30630619task3"
    "slurm30626422task2"
    "slurm30626422task1"
)
#    "factorized-from-pretrained"
#    "factorized-from-pretrained"
#    "factorized-from-pretrained"
#    "factorized-from-pretrained"
#    "factorized-from-pretrained"
#    "pretrained-only"
#    "pretrained-only"
#    "factorized-from-pretrained"
MODEL_MODES=(
   "factorized-from-pretrained"
   "factorized-from-pretrained"
   "pretrained-only"
   "factorized-from-pretrained"
)
source activate methylseqnet
python single_model_atlas_longread.py --model-identifier ${MODEL_IDENTIFIERS[$SLURM_ARRAY_TASK_ID]} --gpus 1 --mode ${MODEL_MODES[$SLURM_ARRAY_TASK_ID]} --checkpoint-type best