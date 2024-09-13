import torch
from torch import nn
from methylseqnet.methylseqnn import MethylSeqNN
from methylseqnet.trainer import Trainer
import json
from pathlib import Path
from methylseqnet.dataset import CustomH5Dataset
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import gin
from collections import defaultdict
import re
# import psutil
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_for_eval(model_path: str | Path):
    
    saved_model = torch.load(str(model_path)+'_state.pth',map_location=torch.device(device))
    operative_config_str = saved_model['gin_file']
    gin.parse_config(operative_config_str)
    # with open(str(model_path)+'_meta.json', 'r') as f:
    #     metadata_list = json.load(f)
    # hyperparams = metadata_list[0]
    model = MethylSeqNN()
    model.load_state_dict(saved_model['model_state_dict'])
    model.eval()
    return model

def get_attribute_value(model_path: str | Path, attribute: str):
    pattern = rf'{attribute} = (.+)'
    saved_model = torch.load(str(model_path)+'_state.pth',map_location=torch.device(device))
    operative_config_str = saved_model['gin_file']
    match = re.search(pattern, operative_config_str)
    if match:
        return match.group(1).strip().strip('\'"')
    return ''

def run_whole_dataset(
    model_path: str | Path,
    dataset_path: str | Path,
    batch_size: int=64,
    track_index: int|None = None,
    layer_name: str | None = None,
):
    """
    This function takes an h5 dataset (could be train, valid, test, etc) and runs through inference end-to-end for all
    samples, with a specified model (which must include it's own hyperparamter gin str). Batch size goes to a reasonable
    default. 
    """ 
    model = load_for_eval(model_path).to(device)

    # If layer_name is specified, register a hook to capture its activations
    if layer_name:
        def hook(module, input, output):
            act = output.cpu().detach().numpy()
            # TODO: save activations to disk; can't store in memory it'll crash
            print(f"Captured activations with shape: {act.shape}")  # Print the shape of activations
        layer = dict(model.named_modules()).get(layer_name)
        if layer is None:
            raise ValueError(f"Layer {layer_name} not found in the model")
        hook_handle = layer.register_forward_hook(hook)
    
    dataset = CustomH5Dataset(dataset_path,batch_size=batch_size)
    dataloader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=1)
    
    targets_list = []
    outputs_list = []
    
    for batch_idx, (inputs, targets, mask) in enumerate(tqdm(dataloader,unit='batch',desc='model passes')):

        targets = targets.permute(0, 2, 1)

        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)

        if mask is not None:
            mask = mask.permute(0, 2, 1)
            mask.to(device)
        else:
            # for code clarity, we make a "fake" mask that is just True everywhere
            # this means we don't need any other if statements to handle None, and
            # it means the later mask application will still squeeze the targets and 
            # outputs even if it doesn't remove any elements
            mask = torch.ones_like(targets, dtype=torch.bool)
            mask.to(device)
        
        # this applies the appropriate masking and reshapes to 1d so we can directly extend the list
        targets = targets[mask]
        outputs = outputs[mask]

        targets_list.extend(targets.cpu().detach().numpy().tolist())
        outputs_list.extend(outputs.cpu().detach().numpy().tolist())

    
    # Remove hook to release memory
    if layer_name:
        hook_handle.remove()
        # Flatten activations list
        activations = np.concatenate(activations, axis=0)
    
    targets = torch.tensor(targets_list).numpy()
    probabilities = torch.sigmoid(torch.tensor(outputs_list)).numpy()
    
    return targets,probabilities

def run_whole_dataset_specified_indices(
    model_path: str | Path,
    dataset_path: str | Path,
    batch_size: int=64,
    track_indices: list=[],
):
    model = load_for_eval(model_path).to(device)
    
    dataset = CustomH5Dataset(dataset_path,batch_size=batch_size)
    dataloader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=1)
    
    targets_dict = defaultdict(list)
    outputs_dict = defaultdict(list)
    
    for inputs, targets, mask in tqdm(dataloader,unit='batch',desc='model passes'):
        
        targets = targets.permute(0, 2, 1)
        
        inputs, targets = inputs.to(device), targets.to(device)
    
        outputs = model(inputs)

        if mask is not None:
            mask = mask.permute(0, 2, 1)
            mask.to(device)
        else:
            # for code clarity, we make a "fake" mask that is just True everywhere
            # this means we don't need any other if statements to handle None, and
            # it means the later mask application will still squeeze the targets and 
            # outputs even if it doesn't remove any elements
            mask = torch.ones_like(targets, dtype=torch.bool)
            mask.to(device)
        
        # We want to subset the batch to 
        # samples where the track_index is unmasked before we get to actually running a 
        # forward pass of the model
        for track_index in track_indices:
            subset_mask = mask[:,track_index,:]
            sample_indices = subset_mask.any(dim=-1)
            track_mask = mask[sample_indices,:,:]
            track_targets = targets[sample_indices,:,:][track_mask]
            track_outputs = outputs[sample_indices,:,:][track_mask]

            targets_dict[track_index].extend(track_targets.cpu().detach().numpy().tolist())
            outputs_dict[track_index].extend(track_outputs.cpu().detach().numpy().tolist())
    return targets_dict,outputs_dict

def load_binned_input_specified_indices(
    dataset_path: str | Path,
    summary_stat: str='cpg_methylation_fraction',
    batch_size: int=64,
    track_indices: list=[],
    bin_size: int=128,
    trim_off_ends: int=384,
):
    """
    summary_stat: the stat to return for each bin. 
        Options:
        -cpg_methylation_fraction (default): the average methylation level of Cs in CG motifs in the bin
        -cpg_count: the number of CpG motifs in the bin (each counts as two, one per strand)
        -gc_content: the fraction of bases that are Gs or Cs in the bin
    """
    dataset = CustomH5Dataset(dataset_path,batch_size=batch_size)
    dataloader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=10)  

    stats_by_index = defaultdict(list)

    pbar = tqdm(dataloader,unit='batch',desc='loading batches')

    for inputs, _, mask in pbar:  
        
        if mask is not None:
            mask = mask.permute(0, 2, 1)
        else:
            # for code clarity, we make a "fake" mask that is just True everywhere
            # this means we don't need any other if statements to handle None, and
            # it means the later mask application will still squeeze the targets and 
            # outputs even if it doesn't remove any elements
            mask = torch.ones_like(targets, dtype=torch.bool)

        # We want to subset the batch to 
        # samples where the track_index is unmasked before we get to actually running a 
        # forward pass of the model
        for track_index in track_indices:
            subset_mask = mask[:,track_index,:]
            sample_indices = subset_mask.any(dim=-1)
            track_inputs = inputs[sample_indices,:,:]

            if summary_stat=='cpg_methylation_fraction':           
                # Define the subarrays for CpG on either strand
                subarray = torch.tensor([[0, 0], [0, 0], [1, 0], [0, 1]]).unsqueeze(0)
    
                sliced_inputs = track_inputs[:,:4, :]
    
                matches = (torch.all(sliced_inputs[:,:, :-1] == subarray[:,:, :1], dim=1) & \
                torch.all(sliced_inputs[:,:, 1:] == subarray[:,:, 1:], dim=1)).unsqueeze(1)
    
                extended_matches = torch.zeros((matches.shape[0], 1, matches.shape[2] + 1), dtype=torch.bool)
                extended_matches[:,:,:-1] += matches
                extended_matches[:,:,1:] += matches
    
                cpg_inputs = track_inputs[:,4:5,:]
    
                cpgs_trimmed = cpg_inputs[:,:,trim_off_ends:-trim_off_ends]
                matches_trimmed = extended_matches[:,:,trim_off_ends:-trim_off_ends]
    
                if cpgs_trimmed.shape[0]>0:
                    new_shape = cpgs_trimmed.shape[:-1] + (-1, bin_size)
                    cpgs_reshaped = cpgs_trimmed.view(new_shape)
                    cpgs_binned = cpgs_reshaped.sum(dim=-1)
                    matches_reshaped = matches_trimmed.view(new_shape)
                    matches_binned = matches_reshaped.sum(dim=-1)
        
                    # Define the value to place for zero-denominator addresses
                    zero_denominator_value = torch.tensor(float(0.5))  # or any other value you prefer
                    
                    # Perform the division safely
                    fractions_binned = torch.where(matches_binned != 0, cpgs_binned / matches_binned, zero_denominator_value)
        
                    stats_by_index[track_index].extend(fractions_binned.view(-1).cpu().detach().numpy().tolist())
            elif summary_stat=='cpg_count':
                # Define the subarrays for CpG on either strand
                subarray = torch.tensor([[0, 0], [0, 0], [1, 0], [0, 1]]).unsqueeze(0)
    
                sliced_inputs = track_inputs[:,:4, :]
    
                matches = (torch.all(sliced_inputs[:,:, :-1] == subarray[:,:, :1], dim=1) & \
                torch.all(sliced_inputs[:,:, 1:] == subarray[:,:, 1:], dim=1)).unsqueeze(1)
    
                extended_matches = torch.zeros((matches.shape[0], 1, matches.shape[2] + 1), dtype=torch.bool)
                extended_matches[:,:,:-1] += matches

                matches_trimmed = extended_matches[:,:,trim_off_ends:-trim_off_ends]

                if matches_trimmed.shape[0]>0:
                    new_shape = matches_trimmed.shape[:-1] + (-1, bin_size)
                    matches_reshaped = matches_trimmed.view(new_shape)
                    matches_binned = matches_reshaped.sum(dim=-1)
                    stats_by_index[track_index].extend(matches_binned.view(-1).cpu().detach().numpy().tolist())
            elif summary_stat=='gc_content':
                matches = track_inputs[:,2:3,:] + track_inputs[:,3:4,:]
                matches_trimmed = matches[:,:,trim_off_ends:-trim_off_ends]
                if matches_trimmed.shape[0]>0:
                    new_shape = matches_trimmed.shape[:-1] + (-1, bin_size)
                    matches_reshaped = matches_trimmed.view(new_shape)
                    matches_binned = matches_reshaped.sum(dim=-1)
                    stats_by_index[track_index].extend((matches_binned.view(-1).cpu().detach().numpy()/bin_size).tolist())
                

    return stats_by_index
        