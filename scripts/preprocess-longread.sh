#!/bin/bash
#SBATCH --job-name=preprocess-longread-pbcpg
#SBATCH --account=fc_streets
#SBATCH --partition=savio3
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --output=/clusterfs/nilah/oberon/lightning/sbatch_logs/preprocess_methylseqnet_%A.out
#SBATCH --error=/clusterfs/nilah/oberon/lightning/sbatch_logs/preprocess_methylseqnet_%A.err

source activate methylseqnet

python ../methylseqnet/preprocess.py --config ../configs/preprocess/preprocess_borzoi_longread_allchr_pbcpg.gin
python ../methylseqnet/preprocess.py --config ../configs/preprocess/preprocess_borzoi_longread_chrX_pbcpg.gin
python ../methylseqnet/preprocess.py --config ../configs/preprocess/preprocess_borzoi_longread_allchr.gin
python ../methylseqnet/preprocess.py --config ../configs/preprocess/preprocess_borzoi_longread_chrX.gin