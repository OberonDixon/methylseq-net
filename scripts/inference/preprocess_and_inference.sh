#!/bin/bash
source activate methylseqnet
python ../../methylseqnet/preprocess.py --config ../../configs/preprocess/preprocess_config_motif_insertion.gin
sbatch ../inference/sbatch_run_motif_insertions.sh