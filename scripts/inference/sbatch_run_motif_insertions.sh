#!/bin/bash
#SBATCH --job-name=folders_through_methylseq_2gpu
#SBATCH --account=fc_nilah
#SBATCH --partition=savio3_gpu
#SBATCH --qos=savio_lowprio
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --time=10:00:00
#SBATCH --output=/clusterfs/nilah/oberon/lightning/sbatch_logs/predict_methylseqnet_%A_%a.out
#SBATCH --error=/clusterfs/nilah/oberon/lightning/sbatch_logs/predict_methylseqnet_%A_%a.err
#SBATCH --array=0-2
# Command(s) to run:
# Define an array of model paths
MODEL_IDENTIFIERS=(
    "slurm30634711task0"
    "slurm30634711task0"
    "slurm30630619task3"
)
KWARGS_0=(--no-targets --dataset-type synthetic --synthetic-cpg --variable-input-length --center-methyl-frac 0.15)
KWARGS_1=(--no-targets --dataset-type synthetic --synthetic-cpg --variable-input-length --center-methyl-frac 0.85)
KWARGS_2=(--no-targets --dataset-type synthetic --variable-input-length)

case $SLURM_ARRAY_TASK_ID in
    0) KWARGS=("${KWARGS_0[@]}") ;;
    1) KWARGS=("${KWARGS_1[@]}") ;;
    2) KWARGS=("${KWARGS_2[@]}") ;;
esac
source activate methylseqnet
python run_dataset_save_h5.py --model-identifier ${MODEL_IDENTIFIERS[$SLURM_ARRAY_TASK_ID]} "${KWARGS[@]}"