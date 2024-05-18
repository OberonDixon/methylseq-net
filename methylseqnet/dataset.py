import h5py
from torch.utils.data import Dataset
import torch
import numpy as np

class CustomH5Dataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.file_path = file_path
        self.transform = transform
        # Load the entire dataset into memory
        with h5py.File(self.file_path, 'r') as f:
            self.sequence_data = np.array(f['sequence']).astype(np.float16)
            self.target_data = np.array(f['tracks']).astype(bool)
            # with h5py.File(self.file_path, 'r') as f:
            self.length = len(f['sequence'])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # with h5py.File(self.file_path, 'r') as f:
        #     input_data = f['sequence'][idx]
        #     target = f['tracks'][idx]
        input_data = self.sequence_data[idx]
        target = self.target_data[idx]
        
        if self.transform:
            input_data = self.transform(input_data)

        input_data = np.transpose(input_data, (1,0))
        input_data = torch.tensor(input_data, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)

        return input_data, target