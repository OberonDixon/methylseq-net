#!/bin/bash

source activate methylseqnet
# python ../../methylseqnet/peaks.py --dataset-paths \
#     /global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/fold1.h5 \
#     /global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/fold3.h5 \
#     /global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/fold5.h5 \
#     /global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/fold7.h5 \
#     --output-directory /global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/peaks \
#     --label-substrings acinar adipocyte memory "mammary basal epi" cilliated cardiac hepatocyte "killer t" oligodendrocyte neuron \
#     --num-peaks 500 \
#     --min-peak-distance 131072 \
#     --random-seeds 1 2 3 4 5 6 7 8 9 10

python create_motif_insertion_fastas.py \
    /global/scratch/users/arbajwa/scbasset_data/Homo_sapiens_motif_fasta/ \
    /global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/peaks \
    /global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/motif_inserted_peaks_2048 \
    --INPUT_LEN 16384 \
    --SHUFFLE_LEN 2048 \
    --OVERWRITE

# python ../../methylseqnet/preprocessor.py --config ../../configs/preprocessor/preprocessor_config_motif_insertion.gin

# sbatch ../inference/sbatch_run_motif_insertions.sh