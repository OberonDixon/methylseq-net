import torch
from methylseqnet.methylseqnn import MethylSeqNN
import json
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_for_eval(model_path: str | Path):
    
    saved_model = torch.load(str(model_path)+'_state.pth',map_location=torch.device(device))
    with open(str(model_path)+'_meta.json', 'r') as f:
        metadata_list = json.load(f)
    hyperparams = metadata_list[0]
    model = MethylSeqNN(hyperparams)
    model.load_state_dict(saved_model['model_state_dict'])
    model.eval()
    return model
    