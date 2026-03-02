#!/bin/bash
#SBATCH --job-name=write_motif_seqs_methylseqnet
#SBATCH --account=fc_nilah
#SBATCH --partition=savio3_bigmem
#SBATCH --qos=savio_normal
#SBATCH --requeue
#SBATCH --time=4:00:00
#SBATCH --output=/clusterfs/nilah/ayesha/slurm_methylseq/slurm_job_%j.out
#SBATCH --error=/clusterfs/nilah/ayesha/slurm_methylseq/slurm_job_%j.err
#SBATCH --mail-type=ALL  
#SBATCH --mail-user=arbajwa@berkeley.edu


source activate /clusterfs/nilah/ayesha/envs/methylseq
pip install ../../.

python ../../methylseqnet/preprocess.py --config ../../configs/motif_insertion/preprocess_config_atac_peak_motif_insertion_test_524288.gin
