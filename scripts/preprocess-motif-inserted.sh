#!/bin/bash

source activate methylseqnet

python ../methylseqnet/peaks.py --dataset-paths \
    /global/scratch/projects/vector_streetslab/oberon/methylseqnet_manuscript_datasets/atlas/fold0.h5 \
    /global/scratch/projects/vector_streetslab/oberon/methylseqnet_manuscript_datasets/atlas/fold1.h5 \
    /global/scratch/projects/vector_streetslab/oberon/methylseqnet_manuscript_datasets/atlas/fold2.h5 \
    /global/scratch/projects/vector_streetslab/oberon/methylseqnet_manuscript_datasets/atlas/fold5.h5 \
    /global/scratch/projects/vector_streetslab/oberon/methylseqnet_manuscript_datasets/atlas/fold6.h5 \
    /global/scratch/projects/vector_streetslab/oberon/methylseqnet_manuscript_datasets/atlas/fold7.h5 \
    --output-directory /global/scratch/projects/vector_streetslab/oberon/methylseqnet_manuscript_datasets/motif_insertion/peaks/random_celltype_peaks_cpg.05-.15_10cts \
    --label-substrings "" \
    --data-type "ATAC-seq" \
    --peak-threshold 10 \
    --num-peaks 500 \
    --min-peak-distance 131072 \
    --cpg-density-range 0.05 0.15 \
    --cpg-density-window 2048 \
    --random-seeds 1

python ../methylseqnet/motifs.py \
    /global/scratch/users/arbajwa/scbasset_data/Homo_sapiens_motif_fasta/ \
    /global/scratch/projects/vector_streetslab/oberon/methylseqnet_manuscript_datasets/motif_insertion/peaks/random_celltype_peaks_cpg.05-.15_10cts \
    /global/scratch/projects/vector_streetslab/oberon/methylseqnet_manuscript_datasets/motif_insertion/fastas/motif_inserted_random_celltype_cpg.05-.15_10cts_peaks_2048_test \
    --INPUT_LEN 16384 \
    --SHUFFLE_LEN 2048 \
    --OVERWRITE

python ../methylseqnet/preprocess.py --config ../../configs/preprocess/preprocess_motif_insert_alltypes_cpg.05-.15_peaks_10cts.gin