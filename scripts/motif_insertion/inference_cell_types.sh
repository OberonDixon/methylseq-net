#!/bin/bash
#SBATCH --job-name=motif_insertions_through_methylseq
#SBATCH --account=fc_nilah
#SBATCH --partition=savio3_gpu
#SBATCH --qos=savio_lowprio
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --time=20:00:00
#SBATCH --output=/clusterfs/nilah/oberon/lightning/sbatch_logs/inference_methylseqnet_%A_%a.out
#SBATCH --error=/clusterfs/nilah/oberon/lightning/sbatch_logs/inference_methylseqnet_%A_%a.err
#SBATCH --array=50
#SBATCH --exclude=n0386.savio4,n0389.savio4,n0134.savio3,n0135.savio3,n0136.savio3,n0137.savio3,n0138.savio3,n0143.savio3,n0144.savio3,n0145.savio3,n0158.savio3,n0159.savio3,n0160.savio3,n0161.savio3,n0174.savio3,n0175.savio3,n0176.savio3
# Command(s) to run:

CELL_TYPES=(
    "acinar_ATAC-seq"
    "adipocyte_ATAC-seq"
    "cardiac_ATAC-seq"
    "cilliated_ATAC-seq"
    "hepatocyte_ATAC-seq"
    "killer_t_ATAC-seq"
    "mammary_basal_epi_ATAC-seq"
    "memory_ATAC-seq"
    "neuron_ATAC-seq"
    "oligodendrocyte_ATAC-seq"
    "CNhs10859_CAGE-seq"
    "CNhs11327_CAGE-seq"
    "CNhs12338_CAGE-seq"
    "CNhs12340_CAGE-seq"
    "CNhs12494_CAGE-seq"
    "CNhs12498_CAGE-seq"
)

MODEL_IDENTIFIERS=(
    "slurm30634711task0"
    "slurm30634711task0"
    "slurm30634711task0"
    "slurm30634711task0"
    "slurm30634711task0"
    "slurm30630619task3"
)

KWARGS_0=(--no-targets --dataset-type synthetic --synthetic-cpg --variable-input-length --center-methyl-frac 0.03)
KWARGS_1=(--no-targets --dataset-type synthetic --synthetic-cpg --variable-input-length --center-methyl-frac 0.25)
KWARGS_2=(--no-targets --dataset-type synthetic --synthetic-cpg --variable-input-length --center-methyl-frac 0.50)
KWARGS_3=(--no-targets --dataset-type synthetic --synthetic-cpg --variable-input-length --center-methyl-frac 0.75)
KWARGS_4=(--no-targets --dataset-type synthetic --synthetic-cpg --variable-input-length --center-methyl-frac 0.95)
KWARGS_5=(--no-targets --dataset-type synthetic --variable-input-length)

CONFIG_IDX=$((SLURM_ARRAY_TASK_ID % 6))
CELL_IDX=$((SLURM_ARRAY_TASK_ID / 6))

case $CONFIG_IDX in
    0) KWARGS=("${KWARGS_0[@]}") ;;
    1) KWARGS=("${KWARGS_1[@]}") ;;
    2) KWARGS=("${KWARGS_2[@]}") ;;
    3) KWARGS=("${KWARGS_3[@]}") ;;
    4) KWARGS=("${KWARGS_4[@]}") ;;
    5) KWARGS=("${KWARGS_5[@]}") ;;
esac

DATASET_PATH="/global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/motif_inserted_preprocessed/${CELL_TYPES[$CELL_IDX]}_peaks_2048/motif_insertions.h5"

source activate methylseqnet

echo "Running model ${MODEL_IDENTIFIERS[$CONFIG_IDX]} on dataset $DATASET_PATH with arguments: ${KWARGS[@]}"

python ../inference/run_dataset_save_h5.py --model-identifier ${MODEL_IDENTIFIERS[$CONFIG_IDX]} \
    --dataset-path "$DATASET_PATH" \
    "${KWARGS[@]}"