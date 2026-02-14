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
import tempfile
import gin
import os
import inspect
import gc

@gin.register
@gin.configurable
class BaseHDF5Dataset(Dataset):
    def __init__(
        self, 
        file_path, 
        batch_size: int | None = None, 
        transforms: tuple = (), 
        return_specifiers: bool = False,
        datasets: set | None = None,
        max_retries: int = 100, 
        retry_delay: int = 2,
    ):
        """
        args:
            - file_path: a path to an h5 file, or a list of paths to h5 files
            - batch_size: how many samples per batch
            - transforms: unused for this class. present because we want a shared interface between dataset classes
            - return_specifiers: whether to return specifiers
            - datasets: list of dataset names to return. if None, return all datasets in the file
            - max_retries: how many times to retry reading from the h5 file in case of failure
            - retry_delay: how many seconds to wait between retries
        """
        if transforms:
            raise NotImplementedError("The BaseHDF5Dataset class cannot currently handle transforms, or rather, the transforms in transforms.py cannot handle the embeddings tensors appropriately. As of March 4 2025 this is planned for later but is not urgent.")
        
        self.file_paths = file_path if isinstance(file_path, list) else [file_path]
        self.batch_size = batch_size
        self.return_specifiers = return_specifiers
        modified_datasets = datasets if not self.return_specifiers else (datasets | {'specifier'} if datasets is not None else None)
        self.datasets = modified_datasets
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # it is crucial that self.file_path be virtual, because the file will be deleted in the __del__ function
        self.file_path = create_virtual_h5_with_attributes(self.file_paths)
        
        # Check dataset details
        lengths = []
        with h5py.File(self.file_path, 'r') as f:
            if self.datasets is not None:
                for dataset in self.datasets:
                    if dataset not in f:
                        raise ValueError(f"Dataset '{dataset}' not found in file {self.file_path}. Available datasets: {list(f.keys())}.")
                    else:
                        lengths.append(len(f[dataset]))
            else:
                for dataset in f.keys():
                    lengths.append(len(f[dataset]))
        if len(set(lengths)) != 1:
            raise ValueError(f"Datasets in {self.file_path} do not all have the same length: found lengths {lengths} for {list(f.keys())}.")
        self.length = lengths[0]
        
    def __len__(self):
        if self.batch_size is not None:
            return (self.length + self.batch_size -1) // self.batch_size
        else:
            return self.length

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size if self.batch_size else idx
        end_idx = min(start_idx + self.batch_size, self.length) if self.batch_size else idx + 1
        for attempt in range(self.max_retries):
            try:
                with h5py.File(self.file_path, 'r') as f:
                    sample = {}
                    if self.datasets is not None:
                        datasets_to_use = self.datasets
                    else:
                        datasets_to_use = f.keys()
                    for dataset in datasets_to_use:
                        if dataset == "specifier":
                            sample[dataset] = f[dataset].asstr()[start_idx:end_idx]
                        else:
                            dataset_tensor = torch.tensor(f[dataset][start_idx:end_idx], dtype=torch.float32)
                            if self.batch_size is None:
                                dataset_tensor = dataset_tensor.squeeze(0)
                            sample[dataset] = dataset_tensor
                    if 'specifier' not in sample and self.return_specifiers:
                        sample['specifier'] = np.array(['' for _ in range(sample[list(f.keys())[0]].shape[0])])
                return sample
            except OSError as e:
                if attempt<self.max_retries-1:
                    print(f"Attempt {attempt + 1} failed with error: {e}. Retrying in {self.retry_delay} seconds.", file=sys.stderr)
                    time.sleep(self.retry_delay)
                else:
                    print(f"Max retries exceeded. Failed to read from HDF5 file: {self.file_path}", file=sys.stderr)
                    raise  # Re-raise the last caught exception

    def get_config(self):
        with h5py.File(self.file_path, 'r') as f:
            gin_config_str = f.attrs['gin_config']
            return gin_config_str

    def get_io_mappings_str(self):
        with h5py.File(self.file_path,'r') as f:
            try:
                io_mappings_str = f.attrs['io_mappings']
            except:
                raise Exception(f"Could not find io_mappings in file for {self.file_path}")
            return io_mappings_str

    def get_io_mappings_df(self):
        try:
            io_mappings_str = self.get_io_mappings_str()
            return pd.read_csv(StringIO(io_mappings_str),sep='\t')
        except:
            return pd.DataFrame() 

    def __del__(self):
        # Cleanup the temporary file when the object is destroyed
        if hasattr(self, 'file_path') and self.file_path and os.path.exists(self.file_path):
            os.remove(self.file_path)

@gin.register
@gin.configurable
class MultiMethylDataset(BaseHDF5Dataset):
    def __init__(self, file_path, batch_size=None, transforms=(), return_specifiers=False, max_retries=100, retry_delay=2):
        self.transforms = [transform() for transform in transforms]
        super().__init__(
            file_path=file_path,
            batch_size=batch_size,
            transforms=(),   # base class raises on non-empty; transforms handled above
            return_specifiers=return_specifiers,
            datasets=['sequence', 'methylation', 'tracks', 'mask'],
            max_retries=max_retries,
            retry_delay=retry_delay,
        )

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size if self.batch_size else idx
        end_idx = min(start_idx + self.batch_size, self.length) if self.batch_size else idx + 1
        for attempt in range(self.max_retries):
            try:
                with h5py.File(self.file_path, 'r', rdcc_nbytes=1024*1024, rdcc_nslots=1) as f:
                    sequence_np = f['sequence'][start_idx:end_idx]
                    methylation_np = f['methylation'][start_idx:end_idx]
                    target_np = f['tracks'][start_idx:end_idx]
                    mask_np = f['mask'][start_idx:end_idx]
                    specifiers = f['specifier'].asstr()[start_idx:end_idx]
                    if self.batch_size is None:
                        specifiers = specifiers[0]

                    # Backward compatibility: expand dimensions if needed
                    if sequence_np.ndim == 3:
                        sequence_np = sequence_np[:, np.newaxis, :, :]
                        target_np = target_np[:, np.newaxis, :, :]
                        batch_size = methylation_np.shape[0]
                        seq_length = methylation_np.shape[2]
                        num_cell_types = methylation_np.shape[1] // 3
                        methylation_np = methylation_np.reshape(batch_size, num_cell_types, 3, seq_length)
                        methylation_np = methylation_np[:, np.newaxis, :, :, :]
                        mask_np = mask_np[:, np.newaxis, :, :]

                    sequence = torch.tensor(sequence_np, dtype=torch.float32)
                    methylation = torch.tensor(methylation_np, dtype=torch.float32)
                    target = torch.tensor(target_np, dtype=torch.float32)
                    mask = torch.tensor(mask_np, dtype=torch.bool)
                    del sequence_np, methylation_np, target_np, mask_np  # free memory
                            
                for transform in self.transforms:
                    sequence, methylation, target, mask = transform(sequence, methylation, target, mask)

                if self.batch_size is None:
                    sequence = sequence.squeeze(0)
                    methylation = methylation.squeeze(0)
                    target = target.squeeze(0)
                    mask = mask.squeeze(0) if mask is not None else mask
        
                if idx%50==0:
                    gc.collect()
                
                return {
                    'sequence': sequence,
                    'conditioning_state': methylation,
                    'target': target,
                    'mask': mask,
                    'specifier': specifiers,
                }
            except OSError as e:
                if attempt<self.max_retries-1:
                    print(f"Attempt {attempt + 1} failed with error: {e}. Retrying in {self.retry_delay} seconds.", file=sys.stderr)
                    time.sleep(self.retry_delay)
                else:
                    # If all retries fail, print the error and re-raise the last exception
                    print(f"Max retries exceeded. Failed to read from HDF5 file: {self.file_path}", file=sys.stderr)
                    raise  # Re-raise the last caught exception
        
def create_virtual_h5_with_attributes(file_paths):
    """
    Creates a temporary HDF5 file that concatenates all datasets from the provided HDF5 files,
    while preserving both global and dataset-specific attributes.
    """
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.h5')
    temp_file.close()
    output_path = temp_file.name

    # Read dataset names and global attributes from the first file
    with h5py.File(file_paths[0], 'r') as f_first:
        dataset_names = list(f_first.keys())
        global_attrs = dict(f_first.attrs)

    with h5py.File(output_path, 'w') as f_out:
        # Copy global attributes
        for attr_name, attr_value in global_attrs.items():
            f_out.attrs[attr_name] = attr_value

        total_lengths = {name: 0 for name in dataset_names}

        # Compute total lengths for each dataset
        for fp in file_paths:
            with h5py.File(fp, 'r') as f:
                for name in dataset_names:
                    if name in f:
                        total_lengths[name] += f[name].shape[0]

        # Create virtual datasets and copy dataset-specific attributes
        offsets = {name: 0 for name in dataset_names}
        layouts = {}

        for name in dataset_names:
            with h5py.File(file_paths[0], 'r') as f:
                if name in f:
                    shape = f[name].shape
                    dtype = f[name].dtype
                    layouts[name] = h5py.VirtualLayout(shape=(total_lengths[name], *shape[1:]), dtype=dtype)

        # Populate virtual layouts
        for fp in file_paths:
            with h5py.File(fp, 'r') as f:
                for name in dataset_names:
                    if name in f:
                        length = f[name].shape[0]
                        src = h5py.VirtualSource(fp, name, shape=f[name].shape)
                        layouts[name][offsets[name]:offsets[name]+length] = src
                        offsets[name] += length

        # Write virtual datasets and copy dataset attributes
        for name, layout in layouts.items():
            vds = f_out.create_virtual_dataset(name, layout)

            # Copy dataset-specific attributes from the first file containing this dataset
            for fp in file_paths:
                with h5py.File(fp, 'r') as f:
                    if name in f:
                        for attr_name, attr_value in f[name].attrs.items():
                            vds.attrs[attr_name] = attr_value
                        break  # Copy attributes only from the first occurrence

    return output_path

def _pack(values):
    # Filter out None values.
    valid = [v for v in values if v is not None]
    if not valid:
        return None
    elif len(valid) == 1:
        return valid[0]
    else:
        return tuple(valid)