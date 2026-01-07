#!/bin/bash
#SBATCH --job-name=write_motif_seqs_methylseqnet
#SBATCH --account=fc_nilah
#SBATCH --partition=savio3_bigmem
#SBATCH --qos=savio_normal
#SBATCH --requeue
#SBATCH --time=72:00:00
#SBATCH --output=/clusterfs/nilah/ayesha/slurm_methylseq/slurm_job_%j.out
#SBATCH --error=/clusterfs/nilah/ayesha/slurm_methylseq/slurm_job_%j.err
#SBATCH --mail-type=ALL  
#SBATCH --mail-user=arbajwa@berkeley.edu

source activate /clusterfs/nilah/ayesha/envs/methylseq

python create_motif_insertion_fastas.py \
'/global/scratch/users/arbajwa/scbasset_data/Homo_sapiens_motif_fasta/' \
'/clusterfs/nilah/oberon/datasets/atac_atlas/peaks' \
'/global/scratch/users/arbajwa/datasets/atac_atlas' \
--INPUT_LEN 524288 \
--PEAK_LEN 128 \