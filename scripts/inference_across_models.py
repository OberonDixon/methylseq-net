from methylseqnet import dna_io, inference, metrics, trainer, model, io_handlers, preprocessor
import methylseqnet
# from dimelo import load_processed, plot_enrichment_profile, utils
from matplotlib import pyplot as plt
import torch
import numpy as np
from Bio import motifs, Seq
from Bio.motifs.matrix import PositionWeightMatrix
import pysam
from tqdm.auto import tqdm
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, precision_recall_curve, auc, f1_score, roc_curve
from matplotlib.ticker import MultipleLocator
import gin
import pickle
import methylseqnet
import os
from methylseqnet.transforms import *
import torch.distributed as dist

# window_size = 2001

model_paths = [
    '/clusterfs/nilah/oberon/lightning/slurm21396548task0/checkpoints/best-checkpoint.ckpt', # regression scaled mse - seq methyl binary, weight decay, dropout8
    '/clusterfs/nilah/oberon/lightning/slurm21396548task1/checkpoints/best-checkpoint.ckpt', # regression scaled mse - seq methyl 5 channel
    '/clusterfs/nilah/oberon/lightning/slurm21396548task2/checkpoints/best-checkpoint.ckpt', # regression scaled mse - seq only
    '/clusterfs/nilah/oberon/lightning/slurm21440087task3/checkpoints/best-checkpoint.ckpt', # regression poisson - seq only
    '/clusterfs/nilah/oberon/lightning/slurm21440087task2/checkpoints/best-checkpoint.ckpt', # regression poisson - seq methyl 5 channel
    '/clusterfs/nilah/oberon/lightning/slurm21440087task1/checkpoints/best-checkpoint.ckpt', # regression poisson - seq methyl binary, weight decay, dropout8
    '/clusterfs/nilah/oberon/lightning/slurm21440087task0/checkpoints/best-checkpoint.ckpt', # regression poisson - seq methyl 7 channel
    # '/clusterfs/nilah/oberon/lightning/slurm21191096task1/checkpoints/best-checkpoint.ckpt', # classification - seq methyl sum-to-one
    # '/clusterfs/nilah/oberon/lightning/slurm21191096task2/checkpoints/best-checkpoint.ckpt', # classification - seq only 7 epochs
    # '/clusterfs/nilah/oberon/lightning/slurm21197035task0/checkpoints/best-checkpoint.ckpt', # classification - seq methyl binary
    # '/clusterfs/nilah/oberon/lightning/slurm21242970task0/checkpoints/best-checkpoint.ckpt', # classification - seq methyl binary, weight decay, dropout8
    # '/clusterfs/nilah/oberon/lightning/slurm21161432task3/checkpoints/best-checkpoint.ckpt', # classification - seq methyl 5 channel
    # '/global/scratch/users/dixonluinenburg/atlas_datasets/lightning_logs/lightning_logs/2024-09-18_23-08-37/checkpoints/',
    # '/global/scratch/users/dixonluinenburg/atlas_datasets/lightning_logs/lightning_logs/2024-09-19_09-27-19/checkpoints/',
    # '/global/scratch/users/dixonluinenburg/atlas_datasets/lightning_logs/lightning_logs/2024-09-19_09-31-07/checkpoints/',
    # '/global/scratch/users/dixonluinenburg/atlas_datasets/lightning_logs/lightning_logs/2024-09-19_09-32-29/checkpoints/',
    # '/global/scratch/users/dixonluinenburg/atlas_datasets/lightning_logs/lightning_logs/2024-09-19_14-14-00/checkpoints/',
]
# model_paths = [
#     max([os.path.join(directory, f) for f in os.listdir(directory)], key=os.path.getctime)
#     for directory in model_checkpoint_directories
# ]

model_results_dict = {}
test_dataset = '/global/scratch/users/dixonluinenburg/atlas_datasets/regression/test.h5'
for model_path in model_paths:
    name = str(model_path)
    targets_list,outputs_list,rank = inference.run_whole_dataset(
        model_path,
        test_dataset,
        # layers_to_prepend=[SmoothMethylationTransform(window_size=window_size)],
        batch_size=32,
        gpus='auto',
    )
    model_results_dict[name] = targets_list,outputs_list
    # print('rank is',rank)
    # Saving to file
    if rank == 0:
        with open(f'/global/scratch/users/dixonluinenburg/atlas_datasets/regression/models/inference_across_models_2.pkl', 'wb') as file:
            pickle.dump(model_results_dict, file)