import h5py
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm
import pandas as pd
from io import StringIO
import time
import sys

# MEM_LOADER_CHUNKS = 32000

class CustomH5Dataset(Dataset):
    def __init__(self, file_path, batch_size=64, transforms=[], return_specifiers=False, max_retries=100, retry_delay=2):
        self.file_path = file_path
        self.batch_size = batch_size
        self.transforms = [transform() for transform in transforms]
        self.return_specifiers = return_specifiers
        self.max_retries = max_retries
        self.retry_delay = retry_delay
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
        for attempt in range(self.max_retries):
            try:
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
                                # this exists to support legacy datasets and will be obsoleted and removed at some point
                                specifiers = f['region'][start_idx:end_idx]
                            except:
                                # adjust this line when the region option is removed
                                raise ValueError('Dataset contains neither "specifier" nor "region". Consider running with return_specifiers=False')
                            
                
                for transform in self.transforms:
                    input_data, target, mask = transform(input_data, target, mask)
        
                if self.return_specifiers:
                    return input_data, target, mask, specifiers
                else:
                    return input_data, target, mask
            except OSError as e:
                if attempt<self.max_retries-1:
                    print(f"Attempt {attempt + 1} failed with error: {e}. Retrying in {self.retry_delay} seconds.", file=sys.stderr)
                    time.sleep(self.retry_delay)
                else:
                    # If all retries fail, print the error and re-raise the last exception
                    print(f"Max retries exceeded. Failed to read from HDF5 file: {self.file_path}", file=sys.stderr)
                    raise  # Re-raise the last caught exception

    def get_config(self):
        with h5py.File(self.file_path, 'r') as f:
            gin_config_str = f.attrs['gin_config']
            return gin_config_str

    def get_io_mappings_str(self):
        with h5py.File(self.file_path,'r') as f:
            io_mappings_str = f.attrs['io_mappings']
            return io_mappings_str

    def get_io_mappings_df(self):
        try:
            io_mappings_str = self.get_io_mappings_str()
            pd.read_csv(StringIO(io_mappings_str),sep='\t')
        except:
            return pd.DataFrame()
        return 