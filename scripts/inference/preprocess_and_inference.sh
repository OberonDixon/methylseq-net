#!/bin/bash
source activate methylseqnet
python ../../methylseqnet/preprocessor.py --config ../../configs/preprocessor/preprocessor_config_motif_insertion.gin
sbatch ../inference/sbatch_run_motif_insertions.sh