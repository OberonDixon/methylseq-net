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

# python ../../methylseqnet/peaks.py --dataset-paths \
#     /global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/fold1.h5 \
#     /global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/fold3.h5 \
#     /global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/fold5.h5 \
#     /global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/fold7.h5 \
#     --output-directory /global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/peaks_cage \
#     --label-substrings CNhs12494 CNhs11382 CNhs11327 CNhs12498 CNhs12340 CNhs10859 CNhs12338 \
#     --data-type "CAGE-seq" \
#     --num-peaks 500 \
#     --min-peak-distance 131072 \
#     --random-seeds 1 2 3 4 5 6 7

# python ../../methylseqnet/peaks.py --dataset-paths \
#     /global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/fold1.h5 \
#     /global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/fold3.h5 \
#     /global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/fold5.h5 \
#     /global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/fold7.h5 \
#     --output-directory /global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/random_sites \
#     --label-substrings none \
#     --data-type none \
#     --peak-threshold 0 \
#     --num-peaks 500 \
#     --min-peak-distance 131072 \
#     --random-seeds 1

# python ../../methylseqnet/peaks.py --dataset-paths \
#     /global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/fold1.h5 \
#     /global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/fold3.h5 \
#     /global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/fold5.h5 \
#     /global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/fold7.h5 \
#     --output-directory /global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/random_celltype_peaks_5cts \
#     --label-substrings "" \
#     --data-type "ATAC-seq" \
#     --peak-threshold 5 \
#     --num-peaks 500 \
#     --min-peak-distance 131072 \
#     --random-seeds 1

# python create_motif_insertion_fastas.py \
#     /global/scratch/users/arbajwa/scbasset_data/Homo_sapiens_motif_fasta/ \
#     /global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/random_celltype_peaks_5cts \
#     /global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/motif_inserted_random_celltype_5cts_peaks_2048 \
#     --INPUT_LEN 16384 \
#     --SHUFFLE_LEN 2048 \
#     --OVERWRITE

# python ../../methylseqnet/preprocessor.py --config ../../configs/preprocessor/motif_inserts_by_cell_type/preprocessor_config_motif_insert_alltypes_peaks_5cts.gin

# python ../../methylseqnet/peaks.py --dataset-paths \
#     /global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/fold1.h5 \
#     /global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/fold3.h5 \
#     /global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/fold5.h5 \
#     /global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/fold7.h5 \
#     --output-directory /global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/random_celltype_peaks_10cts \
#     --label-substrings "" \
#     --data-type "ATAC-seq" \
#     --peak-threshold 10 \
#     --num-peaks 500 \
#     --min-peak-distance 131072 \
#     --random-seeds 1

# python create_motif_insertion_fastas.py \
#     /global/scratch/users/arbajwa/scbasset_data/Homo_sapiens_motif_fasta/ \
#     /global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/random_celltype_peaks_10cts \
#     /global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/motif_inserted_random_celltype_10cts_peaks_2048 \
#     --INPUT_LEN 16384 \
#     --SHUFFLE_LEN 2048 \
#     --OVERWRITE

# python ../../methylseqnet/preprocessor.py --config ../../configs/preprocessor/motif_inserts_by_cell_type/preprocessor_config_motif_insert_alltypes_peaks_10cts.gin

# python create_motif_insertion_fastas.py \
#     /global/scratch/users/arbajwa/scbasset_data/Homo_sapiens_motif_fasta/ \
#     /global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/random_celltype_peaks_10cts \
#     /global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/motif_inserted_random_celltype_10cts_peaks_2048 \
#     --INPUT_LEN 16384 \
#     --SHUFFLE_LEN 2048 \
#     --OVERWRITE

# python ../../methylseqnet/preprocessor.py --config ../../configs/preprocessor/motif_inserts_by_cell_type/preprocessor_config_motif_insert_alltypes_peaks_10cts.gin

# python ../../methylseqnet/peaks.py --dataset-paths \
#     /global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/fold1.h5 \
#     /global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/fold3.h5 \
#     /global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/fold5.h5 \
#     /global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/fold7.h5 \
#     --output-directory /global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/random_celltype_peaks_cpg.05_10cts \
#     --label-substrings "" \
#     --data-type "ATAC-seq" \
#     --peak-threshold 10 \
#     --num-peaks 500 \
#     --min-peak-distance 131072 \
#     --cpg-density-range 0.045 0.055 \
#     --cpg-density-window 2048 \
#     --random-seeds 1

# python create_motif_insertion_fastas.py \
#     /global/scratch/users/arbajwa/scbasset_data/Homo_sapiens_motif_fasta/ \
#     /global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/random_celltype_peaks_cpg.05_10cts \
#     /global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/motif_inserted_random_celltype_cpg.05_10cts_peaks_2048 \
#     --INPUT_LEN 16384 \
#     --SHUFFLE_LEN 2048 \
#     --OVERWRITE

# python ../../methylseqnet/preprocessor.py --config ../../configs/preprocessor/motif_inserts_by_cell_type/preprocessor_config_motif_insert_alltypes_cpg.05_peaks_10cts.gin

# python ../../methylseqnet/peaks.py --dataset-paths \
#     /global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/fold1.h5 \
#     /global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/fold3.h5 \
#     /global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/fold5.h5 \
#     /global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/fold7.h5 \
#     --output-directory /global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/random_celltype_peaks_cpg.025_10cts \
#     --label-substrings "" \
#     --data-type "ATAC-seq" \
#     --peak-threshold 10 \
#     --num-peaks 500 \
#     --min-peak-distance 131072 \
#     --cpg-density-range 0.02 0.03 \
#     --cpg-density-window 2048 \
#     --random-seeds 1

# python create_motif_insertion_fastas.py \
#     /global/scratch/users/arbajwa/scbasset_data/Homo_sapiens_motif_fasta/ \
#     /global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/random_celltype_peaks_cpg.025_10cts \
#     /global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/motif_inserted_random_celltype_cpg.025_10cts_peaks_2048 \
#     --INPUT_LEN 16384 \
#     --SHUFFLE_LEN 2048 \
#     --OVERWRITE

# python ../../methylseqnet/preprocessor.py --config ../../configs/preprocessor/motif_inserts_by_cell_type/preprocessor_config_motif_insert_alltypes_cpg.025_peaks_10cts.gin

# python ../../methylseqnet/peaks.py --dataset-paths \
#     /global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/fold1.h5 \
#     /global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/fold3.h5 \
#     /global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/fold5.h5 \
#     /global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/fold7.h5 \
#     --output-directory /global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/random_celltype_peaks_cpg.10_10cts \
#     --label-substrings "" \
#     --data-type "ATAC-seq" \
#     --peak-threshold 10 \
#     --num-peaks 500 \
#     --min-peak-distance 131072 \
#     --cpg-density-range 0.08 0.12 \
#     --cpg-density-window 2048 \
#     --random-seeds 1

# python create_motif_insertion_fastas.py \
#     /global/scratch/users/arbajwa/scbasset_data/Homo_sapiens_motif_fasta/ \
#     /global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/random_celltype_peaks_cpg.10_10cts \
#     /global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/motif_inserted_random_celltype_cpg.10_10cts_peaks_2048 \
#     --INPUT_LEN 16384 \
#     --SHUFFLE_LEN 2048 \
#     --OVERWRITE

# python ../../methylseqnet/preprocessor.py --config ../../configs/preprocessor/motif_inserts_by_cell_type/preprocessor_config_motif_insert_alltypes_cpg.10_peaks_10cts.gin

# python ../../methylseqnet/peaks.py --dataset-paths \
#     /global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/fold1.h5 \
#     /global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/fold3.h5 \
#     /global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/fold5.h5 \
#     /global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/fold7.h5 \
#     --output-directory /global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/random_celltype_peaks_cpg.05-.15_10cts \
#     --label-substrings "" \
#     --data-type "ATAC-seq" \
#     --peak-threshold 10 \
#     --num-peaks 500 \
#     --min-peak-distance 131072 \
#     --cpg-density-range 0.05 0.15 \
#     --cpg-density-window 2048 \
#     --random-seeds 1

python create_motif_insertion_fastas.py \
    /global/scratch/users/arbajwa/scbasset_data/Homo_sapiens_motif_fasta/ \
    /global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/random_celltype_peaks_cpg.05-.15_10cts \
    /global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/motif_inserted_random_celltype_cpg.05-.15_10cts_peaks_2048 \
    --INPUT_LEN 16384 \
    --SHUFFLE_LEN 2048 \
    --OVERWRITE

python ../../methylseqnet/preprocessor.py --config ../../configs/preprocessor/motif_inserts_by_cell_type/preprocessor_config_motif_insert_alltypes_cpg.05-.15_peaks_10cts.gin

# python ../../methylseqnet/peaks.py --dataset-paths \
#     /global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/fold1.h5 \
#     /global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/fold3.h5 \
#     /global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/fold5.h5 \
#     /global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/fold7.h5 \
#     --output-directory /global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/random_celltype_peaks_cpg0-.05_10cts \
#     --label-substrings "" \
#     --data-type "ATAC-seq" \
#     --peak-threshold 10 \
#     --num-peaks 500 \
#     --min-peak-distance 131072 \
#     --cpg-density-range 0.0 0.05 \
#     --cpg-density-window 2048 \
#     --random-seeds 1

python create_motif_insertion_fastas.py \
    /global/scratch/users/arbajwa/scbasset_data/Homo_sapiens_motif_fasta/ \
    /global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/random_celltype_peaks_cpg0-.05_10cts \
    /global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/motif_inserted_random_celltype_cpg0-.05_10cts_peaks_2048 \
    --INPUT_LEN 16384 \
    --SHUFFLE_LEN 2048 \
    --OVERWRITE

python ../../methylseqnet/preprocessor.py --config ../../configs/preprocessor/motif_inserts_by_cell_type/preprocessor_config_motif_insert_alltypes_cpg0-.05_peaks_10cts.gin

# python create_motif_insertion_fastas.py \
#     /global/scratch/users/arbajwa/scbasset_data/Homo_sapiens_motif_fasta/ \
#     /global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/random_celltype_peaks_10cts \
#     /global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/motif_inserted_random_celltype_10cts_peaks_2048 \
#     --INPUT_LEN 16384 \
#     --SHUFFLE_LEN 2048 \
#     --OVERWRITE

# python ../../methylseqnet/preprocessor.py --config ../../configs/preprocessor/motif_inserts_by_cell_type/preprocessor_config_motif_insert_alltypes_peaks_10cts.gin

# python ../../methylseqnet/peaks.py --dataset-paths \
#     /global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/fold1.h5 \
#     /global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/fold3.h5 \
#     /global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/fold5.h5 \
#     /global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/fold7.h5 \
#     --output-directory /global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/random_celltype_peaks_15cts \
#     --label-substrings "" \
#     --data-type "ATAC-seq" \
#     --peak-threshold 15 \
#     --num-peaks 500 \
#     --min-peak-distance 131072 \
#     --random-seeds 1

# python create_motif_insertion_fastas.py \
#     /global/scratch/users/arbajwa/scbasset_data/Homo_sapiens_motif_fasta/ \
#     /global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/random_celltype_peaks_15cts \
#     /global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/motif_inserted_random_celltype_15cts_peaks_2048 \
#     --INPUT_LEN 16384 \
#     --SHUFFLE_LEN 2048 \
#     --OVERWRITE

# python ../../methylseqnet/preprocessor.py --config ../../configs/preprocessor/motif_inserts_by_cell_type/preprocessor_config_motif_insert_alltypes_peaks_15cts.gin

# python create_motif_insertion_fastas.py \
#     /global/scratch/users/arbajwa/scbasset_data/Homo_sapiens_motif_fasta/ \
#     /global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/random_sites \
#     /global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/motif_inserted_random_sites_2048 \
#     --INPUT_LEN 16384 \
#     --SHUFFLE_LEN 2048 \
#     --OVERWRITE

# python create_motif_insertion_fastas.py \
#     /global/scratch/users/arbajwa/scbasset_data/Homo_sapiens_motif_fasta/ \
#     /global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/peaks_cage \
#     /global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/motif_inserted_peaks_2048 \
#     --INPUT_LEN 16384 \
#     --SHUFFLE_LEN 2048 \
#     --OVERWRITE

# python ../../methylseqnet/preprocessor.py --config ../../configs/preprocessor/motif_inserts_by_cell_type/preprocessor_config_motif_insert_nonpeaks.gin

# sbatch ../inference/sbatch_run_motif_insertions.sh