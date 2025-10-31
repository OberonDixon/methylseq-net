import bisect
import random

import pandas as pd

from torch.utils.data import Dataset

class MultiKeyDataset(Dataset):
    """
    Wraps multiple datasets with keys, supporting both random sampling and concatenation.
    
    Modes:
    - sample_with_replacement=True: Random sampling for training (ignores actual idx)
    - sample_with_replacement=False: Sequential concatenation for val/predict
    """
    
    def __init__(self, dataset_dict, sample_with_replacement=False, 
                 epoch_size=None, weights=None):
        """
        Args:
            dataset_dict: Dict of {key: dataset}
            sample_with_replacement: If True, randomly sample. If False, concatenate.
            epoch_size: Samples per epoch (only used if sample_with_replacement=True)
            weights: Dict of {key: weight} for sampling (only used if sample_with_replacement=True)
        """
        self.dataset_dict = dataset_dict
        self.keys = list(dataset_dict.keys())
        self.datasets = [dataset_dict[key] for key in self.keys]
        self.sample_with_replacement = sample_with_replacement
        
        # Calculate cumulative sizes for concatenation mode
        self.cumulative_sizes = self._cumsum([len(d) for d in self.datasets])
        
        if sample_with_replacement:
            # Random sampling mode
            self.epoch_size = epoch_size or self.cumulative_sizes[-1]
            if weights is None:
                weights = {key: 1.0 for key in self.keys}
            self.weights = [weights.get(key, 1.0) for key in self.keys]
        else:
            # Concatenation mode - epoch_size cannot be set manually
            if epoch_size is not None:
                raise ValueError("epoch_size cannot be set when sample_with_replacement is False")
            self.epoch_size = self.cumulative_sizes[-1] if self.cumulative_sizes else 0
    
    @staticmethod
    def _cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            r.append(e + s)
            s += e
        return r
    
    def __len__(self):
        return self.epoch_size
    
    def __getitem__(self, idx):
        if self.sample_with_replacement:
            # Random sampling mode - ignore idx
            chosen_key = random.choices(self.keys, weights=self.weights, k=1)[0]
            dataset_idx = self.keys.index(chosen_key)
            sample_idx = random.randint(0, len(self.datasets[dataset_idx]) - 1)
        else:
            # Concatenation mode - use idx deterministically
            dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
            if dataset_idx == 0:
                sample_idx = idx
            else:
                sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        
        sample = self.datasets[dataset_idx][sample_idx]
        sample['dataset_key'] = self.keys[dataset_idx]
        
        return sample

    def get_io_mappings_str(self):
        dfs = []
        channel_offset = 0
        cell_type_offset = 0
        for key, dataset in self.dataset_dict.items():
            df = dataset.get_io_mappings_df()
            df.insert(0, 'dataset_key', key)
            df.insert(1, 'absolute_channel', df['channel'] + channel_offset)
            df.insert(2, 'absolute_cell_type', df['cell_type'] + cell_type_offset)
            dfs.append(df)
            channel_offset += df['channel'].max() + 1
            cell_type_offset += df['cell_type'].max() + 1
        combined_df = pd.concat(dfs, ignore_index=True)
        return combined_df.to_csv(sep='\t', index=False)