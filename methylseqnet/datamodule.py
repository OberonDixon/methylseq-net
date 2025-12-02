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
            epoch_size: 
                - If sample_with_replacement=True: int, samples per epoch
                - If sample_with_replacement=False: tuple of ints (same length as dataset_dict),
                specifying number of samples to take from the beginning of each dataset
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
            if epoch_size is not None and not isinstance(epoch_size, int):
                raise ValueError("epoch_size must be an int when sample_with_replacement is True")
            self.epoch_size = epoch_size or self.cumulative_sizes[-1]
            if weights is None:
                weights = {key: 1.0 for key in self.keys}
            self.weights = [weights.get(key, 1.0) for key in self.keys]
            self.dataset_sizes = None  # Not used in sampling mode
        else:
            # Concatenation mode
            if epoch_size is not None:
                if not isinstance(epoch_size, tuple):
                    raise ValueError("epoch_size must be a tuple when sample_with_replacement is False")
                if len(epoch_size) != len(self.keys):
                    raise ValueError(f"epoch_size tuple must have {len(self.keys)} elements, got {len(epoch_size)}")
                # Validate that each size doesn't exceed dataset length
                for i, (key, size) in enumerate(zip(self.keys, epoch_size)):
                    if size > len(self.datasets[i]):
                        raise ValueError(f"epoch_size[{i}] ({size}) exceeds length of dataset '{key}' ({len(self.datasets[i])})")
                self.dataset_sizes = list(epoch_size)
                self.cumulative_sizes = self._cumsum(self.dataset_sizes)
                self.epoch_size = self.cumulative_sizes[-1] if self.cumulative_sizes else 0
            else:
                # Use full dataset lengths
                self.dataset_sizes = [len(d) for d in self.datasets]
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

            # Validation: ensure we're within the allowed range for this dataset
            if self.dataset_sizes and sample_idx >= self.dataset_sizes[dataset_idx]:
                raise IndexError(f"Sample index {sample_idx} out of range for dataset {dataset_idx}")
        
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