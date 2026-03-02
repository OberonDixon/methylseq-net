#!/bin/bash
#SBATCH --job-name=train_methylseqnet_factorized_1gpu
#SBATCH --account=fc_nilah
#SBATCH --partition=savio3_gpu
#SBATCH --qos=savio_lowprio
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --time=48:00:00
#SBATCH --output=/clusterfs/nilah/oberon/lightning/sbatch_logs/train_methylseqnet_%A_%a.out
#SBATCH --error=/clusterfs/nilah/oberon/lightning/sbatch_logs/train_methylseqnet_%A_%a.err
#SBATCH --exclude=n0134.savio3,n0135.savio3,n0136.savio3,n0137.savio3,n0138.savio3,n0143.savio3,n0144.savio3,n0145.savio3,n0158.savio3,n0159.savio3,n0160.savio3,n0161.savio3,n0174.savio3,n0175.savio3,n0176.savio3
#SBATCH --array=10-10 # Specify the range of array jobs (e.g., 0-2 for 3 configs)
# Command(s) to run:
# Define an array of config files
CONFIG_FILES=(
    "../configs/factorized/cell_atlas_tests/borzoi_probe_atlas.gin"
    "../configs/factorized/longread_preprocess_tests/borzoi_factorized_atlas_dual-reps-conv-out-sharedconv.gin"
    "../configs/factorized/longread_preprocess_tests/borzoi_factorized_longread+atlas_dual-reps-conv-out-sharedconv.gin"
    "../configs/factorized/longread_preprocess_tests/borzoi_factorized_longread-pbcpg+atlas_dual-reps-conv-out-sharedconv.gin"
    "../configs/factorized/longread_preprocess_tests/borzoi_factorized_atlas+longread-pbcpg_dual-reps-conv-out-sharedconv.gin"
    "../configs/factorized/longread_preprocess_tests/borzoi_factorized_longread-pbcpg-phasenorm+atlas_dual-reps-conv-out-sharedconv.gin"
    "../configs/factorized/longread_preprocess_tests/borzoi_factorized_longread-pbcpg-phasenorm+atlas_dual-reps-conv-out-sharedconv-2048.gin"
    "../configs/factorized/longread_preprocess_tests/borzoi_factorized_longread-pbcpg-phasenorm+atlas_dual-reps-sharedtrans.gin"
    "../configs/factorized/longread_preprocess_tests/borzoi_factorized_longread-pbcpg-collapse+atlas_dual-reps-conv-out-sharedconv.gin"
    "../configs/factorized/longread_preprocess_tests/borzoi_factorized_longread-pbcpg-collapse+atlas-rebalance_dual-reps-conv-out-sharedconv.gin"
    "../configs/factorized/longread_preprocess_tests/borzoi_factorized_longread-pbcpg-collapse-no5exon+atlas_dual-reps-conv-out-sharedconv.gin"
)

# Get the config file for this array task
CONFIG_FILE=${CONFIG_FILES[$SLURM_ARRAY_TASK_ID]}

UNIQUE_IDENTIFIER="slurm${SLURM_ARRAY_JOB_ID}task${SLURM_ARRAY_TASK_ID}"
source activate methylseqnet
python ../methylseqnet/train.py --config $CONFIG_FILE --unique_identifier $UNIQUE_IDENTIFIER --batch_size 1
