Steps to run this pipeline

### `create_motif_insertion_fastas.{py/sh}`

Creates motif-inserted sequence files for each TF (N trials of each) for each each accessibility peak in each cell type. Also creates endogenous sequence files.

Provide a:
   - PWMS_TOP_DIR (though this won't be used if you simply edit 'transcription_factors.txt', recommended) where all tf pwms are stored
   - PEAKS_TOP_DIR which should contain a dir (for each cell type with cell type specific peaks)
   - PEAKS_OUTPUT_DIR where the fastas will be created (in cell type dirs according to the ones you provided)

### `run_fastas_to_h5_preprocessing.py`
Provide a:
   - Gin config file (containing the preprocessed and output dirs, plue information about the sequence) see the example. Uses the MultiMethylWriter, default (zero) methylation written during preprocessing.

### `run_inference_save_h5.{py/sh}`
Provide a:
   - MODEL_IDENTIFIER
   - optionally, a SYNTHETIC_CPG and DATASET_TYPE (atlas is default as opposed to synthetic)
   - leave NO_TARGETS as True (default) for this analysis.

### `after_inference.ipynb`

Provides an example of how to load and appropriately split/aggregate the predictions using some helper functions.

### Still TODO

better functionality for saving processed and subset predictions files. VEP between endogenous and averaged inserted sequences. Visualizations.
