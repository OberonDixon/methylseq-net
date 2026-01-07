#!/bin/bash

source activate methylseqnet
python ../../methylseqnet/peaks.py --dataset-paths \
    /global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/fold1.h5 \
    /global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/fold3.h5 \
    /global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/fold5.h5 \
    /global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/fold7.h5 \
    --output-directory /global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/peaks \
    --label-substrings acinar adipocyte beta hepatocyte \
    --num-peaks 100 \
    --min-peak-distance 131072 \
    --random-seed 42

python create_motif_insertion_fastas.py \
    /global/scratch/users/arbajwa/scbasset_data/Homo_sapiens_motif_fasta/ \
    /global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/peaks \
    /global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/motif_inserted_peaks \
    --INPUT_LEN 16384 \
    --PEAK_LEN 128 \
    --OVERWRITE

python ../../methylseqnet/preprocessor.py --config ../../configs/preprocessor/preprocessor_config_motif_insertion.gin