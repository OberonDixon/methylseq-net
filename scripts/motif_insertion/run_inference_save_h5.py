from optparse import OptionParser
from methylseqnet.predict import run_dataset_save_h5
from methylseqnet.transforms import InsertSyntheticCpG
from pathlib import Path
from functools import partial

def main():

    usage = 'usage: %prog [options] <MODEL_IDENTIFIER>'
    parser = OptionParser(usage)

    parser.add_option("--NO_TARGETS",
                      dest="NO_TARGETS",
                      action='store_true',
                      help="If set, do not include target tracks in the output H5 files.")
    parser.add_option("--DATASET_TYPE",
                      dest="DATASET_TYPE",
                      choices=['atlas', 'synthetic'],
                      default='atlas',
                      help="Type of dataset to run predict on.")
    parser.add_option("--SYNTHETIC_CPG",
                      dest="SYNTHETIC_CPG",
                      action='store_true',
                      help="If set, add synthetic CpG data.")
    (options, args) = parser.parse_args()

    if len(args) == 1:
        MODEL_IDENTIFIER = args[0]
    else:
        parser.error('Must provide parameter MODEL_IDENTIFIER')

    ## Not sure about the atlas option here???
    if options.DATASET_TYPE == 'atlas':
        dataset_paths = [
            {
                #"atlas":f"/global/scratch/users/dixonluinenburg/atlas_datasets/borzoi-128lzf-multimethyl-bisulfite-atac-cage/fold{fold}.h5",
                "atlas":"/global/scratch/users/arbajwa/datasets/atac_atlas/preprocessed_peaks/Hepatocyte_peaks_test_small/motif_insertions.h5"
            } for fold in range(8)
        ]
    elif options.DATASET_TYPE == 'synthetic':
        dataset_paths = [
            {
                #"all":"/clusterfs/nilah/oberon/datasets/motif_insertion_test/pred.h5",
                "all":"/global/scratch/users/arbajwa/datasets/atac_atlas/preprocessed_peaks/Hepatocyte_peaks_test_small/motif_insertions.h5"
            }
        ]
    
    if options.SYNTHETIC_CPG:
        transforms = (
            partial(
                InsertSyntheticCpG,
                # modify parameters if needed
                center_window_size = 500,
                flank_width = 1000,
                center_cpg_frac = 0.05,
                flanking_cpg_frac = 0.5,
                background_cpg_frac = 0.95,
                offset = 0,
            ),
        )
    else:
        transforms = ()
    
    for dataset_path in dataset_paths:
        dataset_name = Path(list(dataset_path.values())[0]).stem
        dataset_dir = Path(list(dataset_path.values())[0]).parent
        print(f"Running through {dataset_name}.")
        run_dataset_save_h5(
            model_path=f'/clusterfs/nilah/oberon/lightning/{MODEL_IDENTIFIER}/checkpoints/temp-checkpoint.ckpt',
            mode='factorized-from-pretrained',
            dataset_path=dataset_path,
            dataset_type='multimethyl',
            output_path=dataset_dir / MODEL_IDENTIFIER / dataset_name,
            gpus = 1,
            num_workers = 4,
            no_targets = options.NO_TARGETS,
            transforms = transforms,
        )

if __name__ == "__main__":
    main()
