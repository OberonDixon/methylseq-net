#!/bin/bash
#SBATCH --job-name=folds_through_methylseq_2gpu
#SBATCH --account=fc_nilah
#SBATCH --partition=savio4_gpu
#SBATCH --qos=savio_lowprio
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --requeue
#SBATCH --time=24:00:00
#SBATCH --output=/clusterfs/nilah/oberon/lightning/sbatch_logs/inference_methylseqnet_%A_%a.out
#SBATCH --error=/clusterfs/nilah/oberon/lightning/sbatch_logs/inference_methylseqnet_%A_%a.err
#SBATCH --array=0,1,2,3,5 # Specify the range of array jobs (e.g., 0-2 for 3 configs)

# Command(s) to run:
# Define an array of model paths
MODEL_IDENTIFIERS=(
    "slurm30018035task9"
    "slurm30039609task10"
    "slurm30024664task1"
    "slurm29867045task1"
    "slurm29952986task5"
    "slurm29940167task0"
)
MODEL_MODES=(
    "factorized-from-pretrained"
    "factorized-from-pretrained"
    "factorized-from-pretrained"
    "factorized-from-pretrained"
    "factorized-from-pretrained"
    "pretrained-only"
)
source activate methylseqnet
python single_model_atlas_longread.py --model-identifier ${MODEL_IDENTIFIERS[$SLURM_ARRAY_TASK_ID]} --gpus 2 --mode ${MODEL_MODES[$SLURM_ARRAY_TASK_ID]}