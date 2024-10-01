import h5py
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm

# MEM_LOADER_CHUNKS = 32000

class CustomH5Dataset(Dataset):
    def __init__(self, file_path, batch_size=64, transforms=[], return_specifiers=False):
        self.file_path = file_path
        self.batch_size = batch_size
        self.transforms = transforms
        self.return_specifiers = return_specifiers
        # Check dataset details
        with h5py.File(self.file_path, 'r') as f:
            # Determine the length of the dataset
            self.length = len(f['sequence'])
            if len(f['sequence']) != len(f['tracks']):
                raise ValueError(f"sequence and tracks datasets do not line up: {len(f['sequence'])} vs {len(f['tracks'])} entries respectively.")
            if 'mask' in f:
                self.mask = True
                if len(f['tracks']) != len(f['mask']):
                    raise ValueError(f"tracks and mask datasets do not line up: {len(f['tracks'])} vs {len(f['mask'])} entries respectively.")
            else:
                self.mask = False

        
    def __len__(self):
        return (self.length + self.batch_size -1) // self.batch_size

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.length)
        with h5py.File(self.file_path, 'r') as f:
            input_data = f['sequence'][start_idx:end_idx,:,:]
            target = f['tracks'][start_idx:end_idx,:,:]
            input_data = torch.tensor(input_data, dtype=torch.float32)
            target = torch.tensor(target, dtype=torch.float32)
            if self.mask:
                mask = f['mask'][start_idx:end_idx,:,:]
                mask = torch.tensor(mask, dtype=torch.bool)
            else:
                mask = None
            if self.return_specifiers:
                try:
                    specifiers = f['specifier'][start_idx:end_idx]
                except:
                    try: 
                        specifiers = f['region'][start_idx:end_idx]
                    except:
                        raise ValueError('Dataset contains neither "specifier" nor "region". Consider running with return_specifiers=False')
                    
        
        for transform in self.transforms:
            input_data, target, mask = transform(input_data, target, mask)

        if self.return_specifiers:
            return input_data, target, mask, specifiers
        else:
            return input_data, target, mask