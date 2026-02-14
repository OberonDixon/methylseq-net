import torch
from torch import nn
from methylseqnet.model import ConditionedSeqNN
from methylseqnet.trainer import Trainer
from methylseqnet.io_handlers import *
from methylseqnet.dna_io import one_hot_encode_dna
from methylseqnet.datawriter import BigWigWriter
import json
from pathlib import Path
from methylseqnet.dataset import MultiMethylDataset,BaseHDF5Dataset
from methylseqnet.callbacks import HDF5PredictionWriter
from methylseqnet.trainer import MethylSeqDataModule
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import gin
from collections import defaultdict
import re
import methylseqnet
import os
from lightning import Trainer
import pandas as pd
from io import StringIO
import ast
import re
from multiprocessing import Pool

def run_dataset_save_h5(
    model_path: str | Path,
    dataset_path: str | Path | tuple[str,Path],
    mode: str,
    dataset_type: str,
    output_path: str | Path,
    gpus: int = 1,
    num_workers: int = 4,
    no_targets: bool = False,
    supplemental_predict_outputs: set = set(),
    variable_input_length: bool = False,
    **kwargs,
):
    model = methylseqnet.model.ConditionedSeqNN.load_from_checkpoint(model_path)
    model.eval()
    model.mode=mode
    model.supplemental_predict_outputs = supplemental_predict_outputs
    if variable_input_length:
        model.crop_off_sequence = 0
        model.pretrained_seq_model.crop = nn.Identity()
    
    match dataset_type:
        case 'multimethyl':
            data_module = MethylSeqDataModule(
                predict_dataset_file = dataset_path,
                batch_size = 1,
                dataset_class = MultiMethylDataset,
                num_workers = num_workers,
                **kwargs,
            )
        case 'multimethyl-and-embeddings':
            data_module = MethylSeqDataModule(
                predict_dataset_file = dataset_path,
                batch_size = 1,
                dataset_class = MultiDataset,
                num_workers = num_workers,
                **kwargs,
            )
        case 'methylseq':
            data_module = MethylSeqDataModule(
                predict_dataset_file = dataset_path,
                batch_size = 1,
                dataset_class = MethylSeqDataset,
                num_workers = num_workers,
                **kwargs,
            )
    
    data_module.setup(stage="predict")
    pred_writer = HDF5PredictionWriter(output_dir=output_path, write_interval="batch", no_targets=no_targets)

    trainer = Trainer(
        accelerator="auto",
        devices=gpus,
        strategy="auto",
        callbacks=[pred_writer],
        logger=False,
    )

    trainer.predict(
        model=model,
        dataloaders=data_module,
        return_predictions=False,
    )

def run_whole_dataset(
    model_path: str | Path,
    dataset_path: str | Path,
    layers_to_prepend: list=[],
    batch_size: int=64,
    track_index: int|None = None,
    layer_name: str | None = None,
    gpus: str = 'auto',
    **kwargs,
):
    """
    This function takes an h5 dataset (could be train, valid, test, etc) and runs through inference end-to-end for all
    samples, with a specified model (which must include it's own hyperparamter gin str). Batch size goes to a reasonable
    default. 
    """ 
    # model = load_for_eval(model_path).to(device)
    

    # If layer_name is specified, register a hook to capture its activations
    # if layer_name:
    #     def hook(module, input, output):
    #         act = output.cpu().detach().numpy()
    #         # TODO: save activations to disk; can't store in memory it'll crash
    #         print(f"Captured activations with shape: {act.shape}")  # Print the shape of activations
    #     layer = dict(model.named_modules()).get(layer_name)
    #     if layer is None:
    #         raise ValueError(f"Layer {layer_name} not found in the model")
    #     hook_handle = layer.register_forward_hook(hook)
    
    model = methylseqnet.model.ConditionedSeqNN.load_from_checkpoint(model_path)

    dataset = MethylSeqDataset(dataset_path,batch_size=batch_size)
    dataloader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=3)
    
    trainer = Trainer(accelerator='gpu',**kwargs)

    layers = list(model.layers)
    for preprend_layer in layers_to_prepend[::-1]:
        layers.insert(0, preprend_layer)
    model.layers = nn.ModuleList(layers)
    
    result = trainer.test(model=model,dataloaders=dataloader)
    targets_list = model.test_targets_list
    outputs_list = model.test_outputs_list

    
    # Remove hook to release memory
    if layer_name:
        hook_handle.remove()
        # Flatten activations list
        activations = np.concatenate(activations, axis=0)
    
    # targets = torch.tensor(targets_list).numpy()
    # probabilities = torch.sigmoid(torch.tensor(outputs_list)).numpy()
    
    return targets_list,outputs_list,trainer.global_rank

def run_whole_dataset_specify_dtype(
    model_path: str | Path,
    dataset_path: str | Path,
    data_types: list,
    layers_to_prepend: list=[],
    batch_size: int=64,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = methylseqnet.model.ConditionedSeqNN.load_from_checkpoint(model_path).to(device)

    dataset = MethylSeqDataset(dataset_path,batch_size=batch_size)
    dataloader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=3)

    layers = list(model.layers)
    for preprend_layer in layers_to_prepend[::-1]:
        layers.insert(0, preprend_layer)
    model.layers = nn.ModuleList(layers)

    io_mappings_df = model.get_io_mappings_df()

    dtype_indices_dict = {}
    for data_type in data_types:
        dtype_indices_dict[data_type] = list(io_mappings_df['channel'][io_mappings_df['data_type']==data_type])

    dtype_targets_dict = defaultdict(list)
    dtype_predictions_dict = defaultdict(list)
    
    for inputs,targets,mask in tqdm(dataloader):

        inputs,targets = inputs.to(device),targets.to(device)
        
        outputs = model(inputs)
        trim_off_targets = 2*model.crop_off_final + (not model.pad_all_layers)*(targets.shape[2]-((inputs.shape[2]-model.receptive_field+model.total_stride)//model.total_stride))
        # trim_off_targets = targets.shape[2]-((inputs.shape[2]-model.receptive_field+model.total_stride)//model.total_stride)
        
        if mask is None:
            # for code clarity, we make a "fake" mask that is just True everywhere
            # this means we don't need any other if statements to handle None, and
            # it means the later mask application will still squeeze the targets and 
            # outputs even if it doesn't remove any elements
            mask = torch.ones_like(targets, dtype=torch.bool)

        mask.to(device)

        mask = mask[:,:,trim_off_targets // 2:-trim_off_targets // 2]
        targets = targets[:,:,trim_off_targets // 2:-trim_off_targets // 2]
        
        for data_type in data_types:
            dtype_mask = mask.clone()
            mask_mask = torch.ones_like(dtype_mask,dtype=torch.bool)
            mask_mask[:,dtype_indices_dict[data_type],:] = False
            dtype_mask[mask_mask] = False
            dtype_targets_dict[data_type].extend(targets[dtype_mask].cpu().detach().tolist())
            dtype_predictions_dict[data_type].extend(outputs[dtype_mask].cpu().detach().tolist())

    return dtype_targets_dict,dtype_predictions_dict

        


def run_one_locus(
    model_path,
    chromosome,
    start,
    end,
    sequence_path,
    methylation_paths,
    label_paths,
    label_index,
):
    model = methylseqnet.model.ConditionedSeqNN.load_from_checkpoint(model_path)
    if Path(sequence_path).suffix in ['.fa','.fasta']:
        sequence_handler = SingleFastaHandler(ref_genome=sequence_path)
    else:
        raise NotImplementedError(f'No method to handle {Path(sequence_path).name} for sequence_paths.')
    if Path(methylation_paths[0]).suffix in ['.bw','.bigwig']:
        cpg_handler = MultiBigWigCpGHandler(bigwig_files=methylation_paths)
    else:
        raise NotImplementedError(f'No method to handle {Path(methylation_paths[0]).name} for methylation_paths.')
    if Path(label_paths[0]).suffix in ['.bw','.bigwig']:
        label_handler = MultiBigWigLabelHandler(bigwig_files=label_paths,label_bin_size=128)
    elif '.bed.gz' in Path(label_paths[0]).name:
        label_handler = MultiBedGzLabelHandler(bedgz_files=label_paths,label_bin_size=128)
    else:
        raise NotImplementedError(f'No method to handle {Path(label_paths[0]).name} for label_paths.')

    sample = {'source':chromosome,'start':start,'end':end}
    sequence = sequence_handler.load_sequence_batch([sample])[0]
    cpg,valid_cpgs = cpg_handler.load_cpg_batch([sample])
    labels = label_handler.load_labels_batch([sample])[0]
    
    inputs = torch.Tensor(np.transpose(one_hot_encode_dna(
        dna_strand=sequence,
        cpg_methylation=cpg[0],
        valid_cpgs=valid_cpgs[0],
        ),
        (1,0))
    ).unsqueeze(0)

    # and thus for bigger receptive field there are fewer prediction bins along the sequence
    trim_off_targets = labels.shape[0]-(
        (inputs.shape[2]-model.receptive_field+model.total_stride)//model.total_stride
    )
    # raise ValueError(f"receptive field {self.receptive_field}, trim off {trim_off_targets}")
    labels = labels[trim_off_targets // 2:-trim_off_targets // 2]

    output = model(inputs)

    mask = torch.full(output.shape,False)
    mask[:,label_index,:] = True

    return labels,output[mask].detach().numpy()
    
def write_channel(channel, writer, genome_channel_data):
    """
    Function to handle writing a single channel to a BigWig file.
    Args:
        channel: Channel name (key in genome_channels_dict).
        writer: BigWigWriter object for this channel.
        genome_channel_data: Data for the channel to be written.
    """
    writer.write_genome(genome_channel_data) 

# Wrapper function to unpack arguments for imap_unordered
def write_channel_wrapper(args):
    return write_channel(*args)

def run_whole_genome_write_methylation(
    model,
    ref_genome,
    output_directory,
    chunk_size=131072,
    early_stop=None,
    trim_off_targets=None,
):
    import pysam

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    bin_size = model.total_stride
    targets_size = chunk_size//bin_size
    if trim_off_targets is None:
        trim_off_targets = 2*model.crop_off_final + (not model.pad_all_layers)*(targets_size-((chunk_size-model.receptive_field+model.total_stride)//model.total_stride))

    io_mappings_df = model.get_io_mappings_df()
    output_file_paths_dict = {}
    for channel,entry in zip(io_mappings_df['cell_type'],io_mappings_df['label_files']):
        parsed_list = ast.literal_eval(entry.replace("PosixPath", ""))
        paths = [Path(p.strip("'")) for p in parsed_list]
    
        # Extract file names
        file_names = [p.stem for p in paths]
        
        # Split file names into components using regex (non-alphanumeric delimiters)
        split_file_names = [re.split(r'[-_.]', name) for name in file_names]
        
        # Find matching components in their original order
        matching_ordered_substrings = split_file_names[0]  # Start with tokens from the first file name
        for tokens in split_file_names[1:]:
            matching_ordered_substrings = [
                substring
                for substring in matching_ordered_substrings
                if substring in tokens
            ]
        
        # Combine matching substrings into a synthetic name
        output_file_paths_dict[channel] = Path(output_directory) / f"Synthetic-{'-'.join(matching_ordered_substrings)}.hg38.bigwig"

    genome_channels_dict = defaultdict(lambda: defaultdict(dict))

    with pysam.FastaFile(ref_genome) as fasta:
        contigs =  [(name, fasta.get_reference_length(name)) for name in fasta.references if '_' not in name]
        bigwig_datawriters_dict = {channel: BigWigWriter(str(path),contigs) for channel,path in output_file_paths_dict.items()}
        for contig,length in tqdm(contigs):
            methylation_predictions = {channel:np.zeros(length) for channel in bigwig_datawriters_dict.keys()}
            motif_sites = {channel:np.zeros(length,dtype=bool) for channel in bigwig_datawriters_dict.keys()}
            for channel in bigwig_datawriters_dict.keys():
                genome_channels_dict[channel][contig]['start']=0 
                genome_channels_dict[channel][contig]['end']=length
            for start in tqdm(range(0,length-chunk_size,chunk_size-trim_off_targets*bin_size),leave=False):

                pred_start = start + trim_off_targets*bin_size//2
                pred_end = start + chunk_size - trim_off_targets*bin_size//2        

                if early_stop and (pred_start > early_stop):
                    continue
                    
                sequence = fasta.fetch(contig,start,start+chunk_size).upper()
                # Convert the sequence into a NumPy array
                seq_array = np.array(list(sequence))[trim_off_targets*bin_size//2:chunk_size - trim_off_targets*bin_size//2]
                cg_mask = (seq_array[:-1] == "C") & (seq_array[1:] == "G")
                # Expand the mask to include both "C" and "G" positions in the motif
                cg_motifs = np.zeros(len(seq_array), dtype=bool)
                cg_motifs[:-1] |= cg_mask  # Mark the "C" positions
                cg_motifs[1:] |= cg_mask   # Mark the "G" positions
                
                input = torch.Tensor(
                    np.transpose(
                        one_hot_encode_dna(dna_strand=sequence),
                        (1,0),
                    )
                ).unsqueeze(0).to(device)
                
                logits = model(input)
                output = torch.sigmoid(logits)
                for channel,writer in bigwig_datawriters_dict.items():
                    unbinned_output = np.repeat(output[0,channel,:].detach().cpu().numpy(),bin_size)
                    methylation_predictions[channel][pred_start:pred_end] = unbinned_output
                    motif_sites[channel][pred_start:pred_end] = cg_motifs

                # clear out the big arrays/tensors to avoid memory issues as we loop through
                del sequence, seq_array, cg_mask, cg_motifs, input, logits, output, unbinned_output
                torch.cuda.empty_cache()

            for channel in genome_channels_dict.keys():
                motif_indices = np.where(motif_sites[channel])[0]  # Indices where CG motifs occur
                filtered_entries = methylation_predictions[channel][motif_indices]  # Filtered values
                filtered_positions = motif_indices  # Map indices to genome positions
                genome_channels_dict[channel][contig]['entries'] = filtered_entries
                genome_channels_dict[channel][contig]['motifs'] = filtered_positions
  
        # Prepare arguments for parallel processing
        args = [
            (channel, writer, genome_channels_dict[channel])
            for channel, writer in bigwig_datawriters_dict.items()
        ]       
        
        # Use multiprocessing Pool for parallel writes
        with Pool(processes=os.cpu_count()) as pool:
            with tqdm(total=len(args), desc="Writing BigWig files") as pbar:
                for _ in pool.imap_unordered(write_channel_wrapper, args):
                    pbar.update(1)
    
# def run_whole_dataset_specified_indices(
#     model_path: str | Path,
#     dataset_path: str | Path,
#     batch_size: int=64,
#     track_indices: list=[],
# ):
#     model = load_for_eval(model_path).to(device)
    
#     dataset = CustomH5Dataset(dataset_path,batch_size=batch_size)
#     dataloader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=1)
    
#     targets_dict = defaultdict(list)
#     outputs_dict = defaultdict(list)
    
#     for inputs, targets, mask in tqdm(dataloader,unit='batch',desc='model passes'):
        
#         targets = targets.permute(0, 2, 1)
        
#         inputs, targets = inputs.to(device), targets.to(device)
    
#         outputs = model(inputs)

#         if mask is not None:
#             mask = mask.permute(0, 2, 1)
#             mask.to(device)
#         else:
#             # for code clarity, we make a "fake" mask that is just True everywhere
#             # this means we don't need any other if statements to handle None, and
#             # it means the later mask application will still squeeze the targets and 
#             # outputs even if it doesn't remove any elements
#             mask = torch.ones_like(targets, dtype=torch.bool)
#             mask.to(device)
        
#         # We want to subset the batch to 
#         # samples where the track_index is unmasked before we get to actually running a 
#         # forward pass of the model
#         for track_index in track_indices:
#             subset_mask = mask[:,track_index,:]
#             sample_indices = subset_mask.any(dim=-1)
#             track_mask = mask[sample_indices,:,:]
#             track_targets = targets[sample_indices,:,:][track_mask]
#             track_outputs = outputs[sample_indices,:,:][track_mask]

#             targets_dict[track_index].extend(track_targets.cpu().detach().numpy().tolist())
#             outputs_dict[track_index].extend(track_outputs.cpu().detach().numpy().tolist())
#     return targets_dict,outputs_dict

# def load_binned_input_specified_indices(
#     dataset_path: str | Path,
#     summary_stat: str='cpg_methylation_fraction',
#     batch_size: int=64,
#     track_indices: list=[],
#     bin_size: int=128,
#     trim_off_ends: int=384,
# ):
#     """
#     summary_stat: the stat to return for each bin. 
#         Options:
#         -cpg_methylation_fraction (default): the average methylation level of Cs in CG motifs in the bin
#         -cpg_count: the number of CpG motifs in the bin (each counts as two, one per strand)
#         -gc_content: the fraction of bases that are Gs or Cs in the bin
#     """
#     dataset = CustomH5Dataset(dataset_path,batch_size=batch_size)
#     dataloader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=10)  

#     stats_by_index = defaultdict(list)

#     pbar = tqdm(dataloader,unit='batch',desc='loading batches')

#     for inputs, _, mask in pbar:  
        
#         if mask is not None:
#             mask = mask.permute(0, 2, 1)
#         else:
#             # for code clarity, we make a "fake" mask that is just True everywhere
#             # this means we don't need any other if statements to handle None, and
#             # it means the later mask application will still squeeze the targets and 
#             # outputs even if it doesn't remove any elements
#             mask = torch.ones_like(targets, dtype=torch.bool)

#         # We want to subset the batch to 
#         # samples where the track_index is unmasked before we get to actually running a 
#         # forward pass of the model
#         for track_index in track_indices:
#             subset_mask = mask[:,track_index,:]
#             sample_indices = subset_mask.any(dim=-1)
#             track_inputs = inputs[sample_indices,:,:]

#             if summary_stat=='cpg_methylation_fraction':           
#                 # Define the subarrays for CpG on either strand
#                 subarray = torch.tensor([[0, 0], [0, 0], [1, 0], [0, 1]]).unsqueeze(0)
    
#                 sliced_inputs = track_inputs[:,:4, :]
    
#                 matches = (torch.all(sliced_inputs[:,:, :-1] == subarray[:,:, :1], dim=1) & \
#                 torch.all(sliced_inputs[:,:, 1:] == subarray[:,:, 1:], dim=1)).unsqueeze(1)
    
#                 extended_matches = torch.zeros((matches.shape[0], 1, matches.shape[2] + 1), dtype=torch.bool)
#                 extended_matches[:,:,:-1] += matches
#                 extended_matches[:,:,1:] += matches
    
#                 cpg_inputs = track_inputs[:,4:5,:]
    
#                 cpgs_trimmed = cpg_inputs[:,:,trim_off_ends:-trim_off_ends]
#                 matches_trimmed = extended_matches[:,:,trim_off_ends:-trim_off_ends]
    
#                 if cpgs_trimmed.shape[0]>0:
#                     new_shape = cpgs_trimmed.shape[:-1] + (-1, bin_size)
#                     cpgs_reshaped = cpgs_trimmed.view(new_shape)
#                     cpgs_binned = cpgs_reshaped.sum(dim=-1)
#                     matches_reshaped = matches_trimmed.view(new_shape)
#                     matches_binned = matches_reshaped.sum(dim=-1)
        
#                     # Define the value to place for zero-denominator addresses
#                     zero_denominator_value = torch.tensor(float(0.5))  # or any other value you prefer
                    
#                     # Perform the division safely
#                     fractions_binned = torch.where(matches_binned != 0, cpgs_binned / matches_binned, zero_denominator_value)
        
#                     stats_by_index[track_index].extend(fractions_binned.view(-1).cpu().detach().numpy().tolist())
#             elif summary_stat=='cpg_count':
#                 # Define the subarrays for CpG on either strand
#                 subarray = torch.tensor([[0, 0], [0, 0], [1, 0], [0, 1]]).unsqueeze(0)
    
#                 sliced_inputs = track_inputs[:,:4, :]
    
#                 matches = (torch.all(sliced_inputs[:,:, :-1] == subarray[:,:, :1], dim=1) & \
#                 torch.all(sliced_inputs[:,:, 1:] == subarray[:,:, 1:], dim=1)).unsqueeze(1)
    
#                 extended_matches = torch.zeros((matches.shape[0], 1, matches.shape[2] + 1), dtype=torch.bool)
#                 extended_matches[:,:,:-1] += matches

#                 matches_trimmed = extended_matches[:,:,trim_off_ends:-trim_off_ends]

#                 if matches_trimmed.shape[0]>0:
#                     new_shape = matches_trimmed.shape[:-1] + (-1, bin_size)
#                     matches_reshaped = matches_trimmed.view(new_shape)
#                     matches_binned = matches_reshaped.sum(dim=-1)
#                     stats_by_index[track_index].extend(matches_binned.view(-1).cpu().detach().numpy().tolist())
#             elif summary_stat=='gc_content':
#                 matches = track_inputs[:,2:3,:] + track_inputs[:,3:4,:]
#                 matches_trimmed = matches[:,:,trim_off_ends:-trim_off_ends]
#                 if matches_trimmed.shape[0]>0:
#                     new_shape = matches_trimmed.shape[:-1] + (-1, bin_size)
#                     matches_reshaped = matches_trimmed.view(new_shape)
#                     matches_binned = matches_reshaped.sum(dim=-1)
#                     stats_by_index[track_index].extend((matches_binned.view(-1).cpu().detach().numpy()/bin_size).tolist())
                

#     return stats_by_index
        