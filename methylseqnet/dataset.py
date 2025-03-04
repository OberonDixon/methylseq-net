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

@gin.register
@gin.configurable
class CustomH5Dataset(Dataset):
    def __init__(self, file_path, batch_size=64, transforms=(), return_specifiers=False, max_retries=100, retry_delay=2):
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
                            specifiers = f['specifier'].asstr()[start_idx:end_idx]
                        except Exception as e:
                            try: 
                                # this exists to support legacy datasets and will be obsoleted and removed at some point
                                specifiers = f['region'][start_idx:end_idx]
                            except:
                                # adjust this line when the region option is removed
                                raise ValueError('Dataset contains neither "specifier" nor "region". Consider running with return_specifiers=False') from e
                            
                
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

@gin.register
@gin.configurable
class MultiMethylDataset(Dataset):
    def __init__(self, file_path, batch_size=64, transforms=(), return_specifiers=False, max_retries=100, retry_delay=2):
        self.file_paths = file_path if isinstance(file_path, list) else [file_path]
        self.batch_size = batch_size
        self.transforms = [transform() for transform in transforms]
        self.return_specifiers = return_specifiers
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # it is crucial that self.file_path be virtual, because the file will be deleted in the __del__ function
        self.file_path = create_virtual_h5_with_attributes(self.file_paths)
        
        # Check dataset details
        with h5py.File(self.file_path, 'r') as f:
            # Determine the length of the dataset
            self.length = len(f['sequence'])
            if len(f['sequence']) != len(f['tracks']) or len(f['tracks']) != len(f['methylation']):
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
                    sequence = f['sequence'][start_idx:end_idx,:,:]
                    methylation = f['methylation'][start_idx:end_idx,:,:]
                    target = f['tracks'][start_idx:end_idx,:,:]
                    sequence = torch.tensor(sequence, dtype=torch.float32)
                    methylation = torch.tensor(methylation, dtype=torch.float32)
                    input_data = torch.cat([sequence,methylation], dim=1)
                    target = torch.tensor(target, dtype=torch.float32)
                    if self.mask:
                        mask = f['mask'][start_idx:end_idx,:,:]
                        mask = torch.tensor(mask, dtype=torch.bool)
                    else:
                        mask = None
                    if self.return_specifiers:
                        try:
                            specifiers = f['specifier'].asstr()[start_idx:end_idx]
                        except Exception as e:
                            # adjust this line when the region option is removed
                            
                            raise ValueError('Dataset does not contain "specifier". Consider running with return_specifiers=False') from e
                            
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
        if os.path.exists(self.file_path):
            os.remove(self.file_path)

@gin.register
@gin.configurable
class EmbeddingsDataset(Dataset):
    def __init__(
        self, 
        file_path, 
        batch_size=64, 
        transforms=(), 
        return_specifiers=False,
        max_retries=100, 
        retry_delay=2,
    ):
        """
        args:
            - file_path: a path to an h5 file, or a list of paths to h5 files
            - batch_size: how many samples per batch
            - transforms: unused for this class. present because we want a shared interface between dataset classes
        """
        self.file_paths = file_path if isinstance(file_path, list) else [file_path]
        self.batch_size = batch_size
        self.return_specifiers = return_specifiers
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # it is crucial that self.file_path be virtual, because the file will be deleted in the __del__ function
        self.file_path = create_virtual_h5_with_attributes(self.file_paths)
        
        # Check dataset details
        with h5py.File(self.file_path, 'r') as f:
            # Determine the length of the dataset
            self.length = len(f['embeddings'])
        
    def __len__(self):
        return (self.length + self.batch_size -1) // self.batch_size

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.length)
        for attempt in range(self.max_retries):
            try:
                with h5py.File(self.file_path, 'r') as f:
                    embeddings = f['embeddings'][start_idx:end_idx,:,:]
                    if self.return_specifiers:
                        try:
                            specifiers = f['specifier'].asstr()[start_idx:end_idx]
                        except:
                            specifiers = np.array(['' for _ in range(embeddings.shape[0])])
                        return torch.tensor(embeddings, dtype=torch.float32), None, None, specifiers
                    else:
                        return torch.tensor(embeddings, dtype=torch.float32), None, None
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

    def __del__(self):
        # Cleanup the temporary file when the object is destroyed
        if os.path.exists(self.file_path):
            os.remove(self.file_path)

@gin.register
@gin.configurable
class MultiDataset(Dataset):
    def __init__(
        self, 
        file_path, 
        dataset_classes, 
        batch_size=64, 
        return_specifiers=False,
        transforms=(), 
        allow_unequal_lengths=True,
        validate_specifiers=True,
    ):
        if not isinstance(file_path,tuple):
            raise TypeError("MultiDataset file_path must be passed as a tuple of file paths corresponding to the dataset_classes.")
        if not isinstance(dataset_classes,tuple):
            raise TypeError("MultiDataset dataset_classes must be pass as a tuple containing class handles.")
        self.datasets = tuple(
            dataset_class(
                file_path=file_path_for_class,
                batch_size=batch_size,
                transforms=transforms,
                return_specifiers=True,
            )
            for dataset_class, file_path_for_class in zip(dataset_classes, file_path)
        )
        self.return_specifiers = return_specifiers
        lengths = [len(dataset) for dataset in self.datasets]
        if not allow_unequal_lengths and len(set(lengths)) > 1:
            raise ValueError(f"All MultiDataset datasets must have the same length; instead found lengths {lengths}. Pass allow_unequal_lengths=True to override.")
        self.validate_specifiers = validate_specifiers
        self.length=min(lengths)
        
    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):  
        results = [dataset[idx] for dataset in self.datasets]
        inputs  = [r[0] for r in results]
        targets = [r[1] for r in results]
        masks   = [r[2] for r in results]
        specifiers = [r[3] for r in results]
        if self.validate_specifiers and len(set([",".join(specifier_list) for specifier_list in specifiers])) > 1:
            raise ValueError(f"Mistmatch between datasets for index {idx}: {specifiers} corresponding to {self.datasets}.")
        if self.return_specifiers:
            return _pack(inputs), _pack(targets), _pack(masks), specifiers[0]
        else:
            return _pack(inputs), _pack(targets), _pack(masks)

    def get_io_mappings_str(self):
        for dataset in self.datasets:
            if hasattr(dataset, "get_io_mappings_str") and callable(getattr(dataset, "get_io_mappings_str")):
                return dataset.get_io_mappings_str()
        raise AttributeError("None of the MultiDataset dataset members can return an io_mappings_str.")

    def get_io_mappings_df(self):
        try:
            io_mappings_str = self.get_io_mappings_str()
            return pd.read_csv(StringIO(io_mappings_str),sep='\t')
        except:
            return pd.DataFrame()         
        
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