#!/bin/bash
#SBATCH --job-name=preprocess_motifs_methylseqnet
#SBATCH --account=fc_nilah
#SBATCH --partition=savio3
#SBATCH --qos=savio_normal
#SBATCH --requeue
#SBATCH --time=1:00:00
#SBATCH --output=/clusterfs/nilah/oberon/lightning/sbatch_logs/preprocess_methylseqnet_%A_%a.out
#SBATCH --error=/clusterfs/nilah/oberon/lightning/sbatch_logs/preprocess_methylseqnet_%A_%a.err
#SBATCH --array=13
#SBATCH --exclude=n0029.savio3,n0048.savio3

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
    "cnhs10859"
    "cnhs11327"
    "cnhs12338"
    "cnhs12340"
    "cnhs12494"
    "cnhs12498"
)

CONFIG_FILE="../../configs/preprocess/motif_inserts_by_cell_type/preprocess_config_motif_insert_${CELL_TYPES[$SLURM_ARRAY_TASK_ID]}.gin"

source activate methylseqnet
python ../../methylseqnet/preprocess.py --config $CONFIG_FILE