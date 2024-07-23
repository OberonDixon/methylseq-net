import torch
from torch import nn
from methylseqnet.methylseqnn import MethylSeqNN
import json
from pathlib import Path
from methylseqnet.dataset import CustomH5Dataset
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import gin

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_for_eval(model_path: str | Path):
    
    saved_model = torch.load(str(model_path)+'_state.pth')
    operative_config_str = saved_model['gin_file']
    gin.parse_config(operative_config_str)
    # with open(str(model_path)+'_meta.json', 'r') as f:
    #     metadata_list = json.load(f)
    # hyperparams = metadata_list[0]
    model = MethylSeqNN()
    model.load_state_dict(saved_model['model_state_dict'])
    model.eval()
    return model

def run_whole_dataset(
    model_path: str | Path,
    dataset_path: str | Path,
    batch_size: int=64,
    track_index: int|None = None,
):
    """
    This function takes an h5 dataset (could be train, valid, test, etc) and runs through inference end-to-end for all
    samples, with a specified model (which must include it's own hyperparamter gin str). Batch size goes to a reasonable
    default. If a track_index is specified AND the dataset has a mask, only the samples where the track index in question
    is at least partly unmasked will be run through the model. Otherwise, all samples will be run.
    """
    
    model = load_for_eval(model_path).to(device)
    
    dataset = CustomH5Dataset(dataset_path,batch_size=batch_size)
    dataloader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=1)
    
    targets_list = []
    outputs_list = []
    
    for inputs, targets, mask in tqdm(dataloader,unit='batch',desc='model passes'):

        targets = targets.permute(0, 2, 1)

        # if mask is not None and track_index is not None, we want to subset the batch to 
        # samples where the track_index is unmasked before we get to actually running a 
        # forward pass of the model
        if mask is not None:
            mask = mask.permute(0, 2, 1)
            if track_index is not None:
                subset_mask = mask[:,track_index,:]
                sample_indices = subset_mask.any(dim=-1)
                inputs = inputs[sample_indices,:,:]
                targets = targets[sample_indices,:,:]
                mask = mask[sample_indices,:,:]
            mask.to(device)
        else:
            # for code clarity, we make a "fake" mask that is just True everywhere
            # this means we don't need any other if statements to handle None, and
            # it means the later mask application will still squeeze the targets and 
            # outputs even if it doesn't remove any elements
            mask = torch.ones_like(targets, dtype=torch.bool)
            mask.to(device)

        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
            
        # this applies the appropriate masking and reshapes to 1d so we can directly extend the list
        targets = targets[mask]
        outputs = outputs[mask]
        targets_list.extend(targets.cpu().detach().numpy().tolist())
        outputs_list.extend(outputs.cpu().detach().numpy().tolist())
    
    targets = torch.tensor(targets_list).numpy()
    probabilities = torch.sigmoid(torch.tensor(outputs_list)).numpy()
    
    return targets,probabilities