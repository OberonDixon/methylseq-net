import h5py
from torch.utils.data import Dataset
import torch
import numpy as np
from tqdm import tqdm

MEM_LOADER_CHUNKS = 32000

class CustomH5Dataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.file_path = file_path
        self.transform = transform
        # Load the entire dataset into memory
        with h5py.File(self.file_path, 'r') as f:
            # Determine the length of the dataset
            self.length = len(f['sequence'])
            
            # Initialize empty numpy arrays with the correct dtype and shape
            sequence_shape = f['sequence'].shape
            target_shape = f['tracks'].shape
            self.sequence_data = np.empty(sequence_shape, dtype=np.float16)
            self.target_data = np.empty(target_shape, dtype=bool)
            
            # Load data in chunks and convert dtype
            for start_idx in tqdm(range(0, self.length, MEM_LOADER_CHUNKS),desc='Loading dataset to memory',unit='chunk'):
                end_idx = min(start_idx + MEM_LOADER_CHUNKS, self.length)
                self.sequence_data[start_idx:end_idx] = f['sequence'][start_idx:end_idx].astype(np.float16)
                self.target_data[start_idx:end_idx] = f['tracks'][start_idx:end_idx].astype(bool)
        # print('sequence footprint')
        # print(self.sequence_data.dtype)
        # print(self.sequence_data.shape)
        # print('itemsize',self.sequence_data.itemsize)
        # print('target footprint')
        # print(self.target_data.dtype)
        # print(self.target_data.shape)
        # print('itemsize',self.target_data.itemsize)

        
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