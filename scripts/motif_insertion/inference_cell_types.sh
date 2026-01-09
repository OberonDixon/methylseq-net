#!/bin/bash
#SBATCH --job-name=motif_insertions_through_methylseq
#SBATCH --account=fc_nilah
#SBATCH --partition=savio4_gpu
#SBATCH --qos=savio_lowprio
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --time=10:00:00
#SBATCH --output=/clusterfs/nilah/oberon/lightning/sbatch_logs/inference_methylseqnet_%A_%a.out
#SBATCH --error=/clusterfs/nilah/oberon/lightning/sbatch_logs/inference_methylseqnet_%A_%a.err
#SBATCH --array=0-29
# Command(s) to run:

CELL_TYPES=(
    "acinar"
    "adipocyte"
    "cardiac"
    "cilliated"
    "hepatocyte"
    "killer_t"
    "mammary_basal_epi"
    "memory"
    "neuron"
    "oligodendrocyte"
)

MODEL_IDENTIFIERS=(
    "slurm30634711task0"
    "slurm30634711task0"
    "slurm30630619task3"
)

KWARGS_0=(--no-targets --dataset-type synthetic --synthetic-cpg --variable-input-length --center-methyl-frac 0.15)
KWARGS_1=(--no-targets --dataset-type synthetic --synthetic-cpg --variable-input-length --center-methyl-frac 0.85)
KWARGS_2=(--no-targets --dataset-type synthetic --variable-input-length)

CONFIG_IDX=$((SLURM_ARRAY_TASK_ID / 10))
CELL_IDX=$((SLURM_ARRAY_TASK_ID % 10))

case $CONFIG_IDX in
    0) KWARGS=("${KWARGS_0[@]}") ;;
    1) KWARGS=("${KWARGS_1[@]}") ;;
    2) KWARGS=("${KWARGS_2[@]}") ;;
esac

DATASET_PATH="/clusterfs/nilah/oberon/datasets/motif_insertion_test/${CELL_TYPES[$CELL_IDX]}_ATAC-seq_peaks/motif_insertions.h5"

source activate methylseqnet

echo "Running model ${MODEL_IDENTIFIERS[$CONFIG_IDX]} on dataset $DATASET_PATH with arguments: ${KWARGS[@]}"

python ../inference/run_dataset_save_h5.py --model-identifier ${MODEL_IDENTIFIERS[$CONFIG_IDX]} \
    --dataset-path "$DATASET_PATH" \
    "${KWARGS[@]}"