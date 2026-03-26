#!/bin/bash
#SBATCH --job-name=predict_motif_insertions
#SBATCH --account=fc_nilah
#SBATCH --partition=savio3_gpu
#SBATCH --qos=a40_gpu3_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A40:1
#SBATCH --requeue
#SBATCH --time=20:00:00
#SBATCH --output=/clusterfs/nilah/oberon/lightning/sbatch_logs/predict_motif_insertions_methylseqnet_%A_%a.out
#SBATCH --error=/clusterfs/nilah/oberon/lightning/sbatch_logs/predict_motif_insertions_methylseqnet_%A_%a.err
#SBATCH --array=1,2,4
#SBATCH --exclude=n0264.savio3,n0005.savio3,n0217.savio3,n0215.savio3,n0130.savio4,n0132.savio4,n0134.savio3,n0135.savio3,n0136.savio3,n0137.savio3,n0138.savio3,n0143.savio3,n0144.savio3,n0145.savio3,n0158.savio3,n0159.savio3,n0160.savio3,n0161.savio3,n0174.savio3,n0175.savio3,n0176.savio3
# Command(s) to run:

CELL_TYPES=(
    "random_celltype_cpg.05-.15_10cts"
)

MODEL_IDENTIFIERS=(
    "slurm32260895task0"
    "slurm32260895task0"
    "slurm32260895task0"
    "slurm32260895task0"
    "slurm32260895task1"
)

KWARGS_0=(--no-targets --dataset-keys all --synthetic-cpg --variable-input-length --center-methyl-frac 0.03)
KWARGS_1=(--no-targets --dataset-keys all --synthetic-cpg --variable-input-length --center-methyl-frac 0.50)
KWARGS_2=(--no-targets --dataset-keys all --synthetic-cpg --variable-input-length --center-methyl-frac 0.95)
KWARGS_3=(--no-targets --dataset-keys all --variable-input-length --true-conditioning-state-weight 0.0)
KWARGS_4=(--no-targets --dataset-keys all --variable-input-length)

CONFIG_IDX=$((SLURM_ARRAY_TASK_ID % 5))
CELL_IDX=$((SLURM_ARRAY_TASK_ID / 5))

case $CONFIG_IDX in
    0) KWARGS=("${KWARGS_0[@]}") ;;
    1) KWARGS=("${KWARGS_1[@]}") ;;
    2) KWARGS=("${KWARGS_2[@]}") ;;
    3) KWARGS=("${KWARGS_3[@]}") ;;
    4) KWARGS=("${KWARGS_4[@]}") ;;
esac

DATASET_PATH="/global/scratch/projects/vector_streetslab/oberon/methylseqnet_manuscript_datasets/motif_insertion/preprocessed/${CELL_TYPES[$CELL_IDX]}_2048/motif_insertions.h5"

source activate methylseqnet-prod

echo "Running model ${MODEL_IDENTIFIERS[$CONFIG_IDX]} on dataset $DATASET_PATH with arguments: ${KWARGS[@]}"

python ../methylseqnet/predict.py --model-identifier ${MODEL_IDENTIFIERS[$CONFIG_IDX]} \
    --dataset-files "$DATASET_PATH" \
    "${KWARGS[@]}"