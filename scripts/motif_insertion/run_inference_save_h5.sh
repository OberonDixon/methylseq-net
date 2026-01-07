#!/bin/bash
#SBATCH --job-name=run_inference_motif_seqs_methylseqnet
#SBATCH --account=fc_nilah
#SBATCH --partition=savio3_gpu
#SBATCH --qos=savio_lowprio
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --requeue
#SBATCH --time=72:00:00
#SBATCH --output=/clusterfs/nilah/ayesha/slurm_methylseq/slurm_job_%j.out
#SBATCH --error=/clusterfs/nilah/ayesha/slurm_methylseq/slurm_job_%j.err
#SBATCH --mail-type=ALL  
#SBATCH --mail-user=arbajwa@berkeley.edu

source activate /clusterfs/nilah/ayesha/envs/methylseq
pip install ../../.

python run_inference_save_h5.py \
--NO_TARGETS \
--DATASET_TYPE 'synthetic' \
--SYNTHETIC_CPG \
'slurm29336762task4'
