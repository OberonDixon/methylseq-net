# methylseq-net

MethylSeqNet is a method for conditioning genomic regulatory activity predictons on epigenetic state, currently in the form of CpG methylation landscape. 

## Reproduce published results

To reproduce results presented in Dixon-Luinenburg et al, 2026, reference reproducibility code in https://github.com/OberonDixon/methylseq-net-reproducibility. This contains examples for key use cases and configurations.

## Running MethylSeqNet

To run the model, you can run `pip install git+https://github.com/OberonDixon/methylseq-net` and use the command line entry points or the exposed submodules. 

### Entry points
#### Preprocess data
Preprocessing scripts take in standard bioinformatic files aligned to a reference. Preprocessing is configured using gin config files.
```
usage: methylseqnet-preprocess [-h] --config CONFIG [--subset SUBSET] [--sequential] [--workers WORKERS]

Run PreprocessingPipeline

options:
  -h, --help         show this help message and exit
  --config CONFIG    Path to the gin config file. No default.
  --subset SUBSET    Subset to process (e.g., train, test, validation, or all). Defaults to all.
  --sequential       Run in sequential mode instead of parallel.
  --workers WORKERS  max_workers across which to parallelize. Defaults to all.
```
#### Training a new model
```
usage: methylseqnet-train [-h] --config CONFIG [--unique-identifier UNIQUE_IDENTIFIER] [--checkpoints-dir CHECKPOINTS_DIR] [--gpus GPUS]
                          [--batch-size BATCH_SIZE] [--max-epochs MAX_EPOCHS] [--samples-per-step SAMPLES_PER_STEP]
                          [--start-from-checkpoint START_FROM_CHECKPOINT] [--no-wandb] [--no-checkpoints] [--no-haplotype-metrics]
                          [--track-gradients-for-modules TRACK_GRADIENTS_FOR_MODULES [TRACK_GRADIENTS_FOR_MODULES ...]] [--seed SEED]
                          [--logging-level {CRITICAL,ERROR,WARNING,INFO,DEBUG,NOTSET}]

Train a ConditionedSeqNN model.

options:
  -h, --help            show this help message and exit
  --config CONFIG       Path to the gin config file.
  --unique-identifier UNIQUE_IDENTIFIER
                        Unique identifier for run.
  --checkpoints-dir CHECKPOINTS_DIR
                        Directory to store output trained checkpoints.
  --gpus GPUS           GPU count for parallelization.
  --batch-size BATCH_SIZE
                        Batch size for dataloader.
  --max-epochs MAX_EPOCHS
                        Maximum number of epochs to train; this is overridden if training stages are defined in the gin config file.
  --samples-per-step SAMPLES_PER_STEP
                        How many samples to process per optimizer step; this is used to calculation gradient accumulation steps internally. If -1,
                        no gradient accumulation is used.
  --start-from-checkpoint START_FROM_CHECKPOINT
                        Unique identifier for a checkpoint from which to restart. Hyperparameter mistmatch may cause errors.
  --no-wandb            Do not save WandB logs.
  --no-checkpoints      Do not save model checkpoints.
  --no-haplotype-metrics
                        If set, enable haplotype-specific metrics logging during training.
  --track-gradients-for-modules TRACK_GRADIENTS_FOR_MODULES [TRACK_GRADIENTS_FOR_MODULES ...]
                        If provided, track gradients for the named module(s). Can be specified multiple times.
  --seed SEED           Random seed for reproducibility.
  --logging-level {CRITICAL,ERROR,WARNING,INFO,DEBUG,NOTSET}
                        Set the logging level
```
#### Predictions with a trained checkpoint
```
usage: methylseqnet-predict [-h] --model-identifier MODEL_IDENTIFIER [--checkpoints-dir CHECKPOINTS_DIR]
                            [--true-conditioning-state-weight TRUE_CONDITIONING_STATE_WEIGHT] [--no-targets] --dataset-keys DATASET_KEYS
                            [DATASET_KEYS ...] --dataset-files DATASET_FILES [DATASET_FILES ...]
                            [--supplemental-outputs [SUPPLEMENTAL_OUTPUTS ...]] [--synthetic-cpg] [--variable-input-length]
                            [--center-methyl-frac CENTER_METHYL_FRAC] [--gpus GPUS] [--num-workers NUM_WORKERS]

Run predictions with a specified model.

options:
  -h, --help            show this help message and exit
  --model-identifier MODEL_IDENTIFIER
                        e.g. slurm24807693task2; will reference checkpoints dir
  --checkpoints-dir CHECKPOINTS_DIR
                        Directory to load trained checkpoints.
  --true-conditioning-state-weight TRUE_CONDITIONING_STATE_WEIGHT
                        If set, override the model's true_conditioning_state_weight with this value for prediction.
  --no-targets          If set, do not include target tracks in the output H5 files.
  --dataset-keys DATASET_KEYS [DATASET_KEYS ...]
                        Dataset keys (e.g., atlas, longread) corresponding to the datasets being predicted on (must match length of --dataset-
                        files)
  --dataset-files DATASET_FILES [DATASET_FILES ...]
                        Paths to dataset H5 files (must match length of --dataset-keys)
  --supplemental-outputs [SUPPLEMENTAL_OUTPUTS ...]
                        Supplemental outputs to include in predictions. Default: ['conditional_seq_rep', 'unconditional_seq_rep',
                        'true_conditioning_state_rep', 'imputed_conditioning_state_rep', 'cpg_density']
  --synthetic-cpg       If set, add synthetic CpG data.
  --variable-input-length
                        If set, sequence length can be any integer multiple of 128 that is >=16384.
  --center-methyl-frac CENTER_METHYL_FRAC
                        Fraction of CpGs methylated in the center window.
  --gpus GPUS           Number of GPUs to use
  --num-workers NUM_WORKERS
                        Number of data loader workers
```
#### Create peak files from preprocessed data
```
usage: methylseqnet-peaks [-h] --dataset-paths DATASET_PATHS [DATASET_PATHS ...] --output-directory OUTPUT_DIRECTORY --label-substrings
                          LABEL_SUBSTRINGS [LABEL_SUBSTRINGS ...] [--data-type DATA_TYPE] [--peak-threshold PEAK_THRESHOLD] [--num-peaks NUM_PEAKS]
                          [--min-peak-distance MIN_PEAK_DISTANCE] [--target-bin-size TARGET_BIN_SIZE]
                          [--cpg-density-range CPG_DENSITY_RANGE CPG_DENSITY_RANGE] [--cpg-density-window CPG_DENSITY_WINDOW]
                          [--random-seeds RANDOM_SEEDS [RANDOM_SEEDS ...]]

Generate peaks bed files from a preprocessed dataset object with io_mappings

options:
  -h, --help            show this help message and exit
  --dataset-paths DATASET_PATHS [DATASET_PATHS ...]
                        Path to the preprocessed dataset object (HDF5 file)
  --output-directory OUTPUT_DIRECTORY
                        Path to save the output BED file
  --label-substrings LABEL_SUBSTRINGS [LABEL_SUBSTRINGS ...]
                        List of substrings to identify label channel from io mappings
  --data-type DATA_TYPE
                        Type of data to process (e.g., 'ATAC-seq', 'ChIP-seq')
  --peak-threshold PEAK_THRESHOLD
                        Threshold to define peaks in preprocessed dataset. If 0, random sites are selected.
  --num-peaks NUM_PEAKS
                        Number of peaks to extract per cell type
  --min-peak-distance MIN_PEAK_DISTANCE
                        Minimum distance between peaks
  --target-bin-size TARGET_BIN_SIZE
                        Size of target bins in the preprocessed dataset
  --cpg-density-range CPG_DENSITY_RANGE CPG_DENSITY_RANGE
                        Optional min and max CpG density range to filter peaks
  --cpg-density-window CPG_DENSITY_WINDOW
                        Window size around peak to compute CpG density
  --random-seeds RANDOM_SEEDS [RANDOM_SEEDS ...]
                        Random seeds per label substring for reproducibility
```
#### Insert motifs into shuffled peaks
```
Usage: methylseqnet-insert-motifs [options] <PWMS_TOP_DIR> <PEAKS_TOP_DIR> <PEAKS_OUTPUT_DIR>

Options:
  -h, --help            show this help message and exit
  --input-len=INPUT_LEN
                        Sequence input length [Default: 524288]
  --shuffle-len=SHUFFLE_LEN
                        Shuffling length around peak center [Default: 128]
  --n=N                 Number of trials for each motif insertion [Default: 5]
  --reference-genome=REFERENCE_GENOME
                        Reference genome fasta path [Default: ]
  --tfs-file=TFS_FILE   Path to text file with TF names (one per line)
                        [Default: transcription_factors.txt]
  --overwrite           Overwrite existing files if present [Default: False]
```