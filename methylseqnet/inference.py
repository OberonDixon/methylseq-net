import torch
from torch import nn
from methylseqnet.methylseqnn import MethylSeqNN
import json
from pathlib import Path
from methylseqnet.dataset import CustomH5Dataset
from tqdm.auto import tqdm
from torch.utils.data import DataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_for_eval(model_path: str | Path):
    
    saved_model = torch.load(str(model_path)+'_state.pth')
    with open(str(model_path)+'_meta.json', 'r') as f:
        metadata_list = json.load(f)
    # hyperparams = metadata_list[0]
    model = MethylSeqNN()
    model.load_state_dict(saved_model['model_state_dict'])
    model.eval()
    return model

def run_whole_dataset(
    model_path: str | Path,
    dataset_path: str | Path,
    batch_size: int=64,
):
    model = load_for_eval(model_path).to(device)
    
    dataset = CustomH5Dataset(dataset_path,batch_size=batch_size)
    dataloader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=1)
    
    targets_list = []
    outputs_list = []
    
    for inputs, targets, mask in tqdm(dataloader,unit='batch',desc='model passes'):
        inputs, targets = inputs.to(device), targets.to(device)

        targets = targets.permute(0, 2, 1)

        outputs = model(inputs)
        if mask is not None:
            mask = mask.permute(0, 2, 1)
            mask.to(device)
            targets = targets[mask]
            outputs = outputs[mask]
        targets_list.extend(targets.cpu().detach().numpy().tolist())
        outputs_list.extend(outputs.cpu().detach().numpy().tolist())
    
    targets = torch.tensor(targets_list).numpy()
    probabilities = torch.sigmoid(torch.tensor(outputs_list)).numpy()
    
    return targets,probabilities