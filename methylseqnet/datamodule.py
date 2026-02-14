import pandas as pd
import gin
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch import LightningDataModule

from methylseqnet.dataset import MultiMethylDataset, CompositeDataset

@gin.configurable
class MethylSeqDataModule(LightningDataModule):
    def __init__(
        self, 
        train_dataset_file=None, 
        validation_dataset_file=None, 
        predict_dataset_file=None,
        batch_size=1, 
        transforms=[], 
        epoch_size=10000,
        val_epoch_size=None,
        dataset_weights=None,
        dataset_class=MultiMethylDataset,
        pow=False, # temporarily restored for backwards compatibility; does nothing
        num_workers=4,
    ):
        super().__init__()
        self.train_dataset_dict = train_dataset_file if isinstance(train_dataset_file, dict) else {"dataset":train_dataset_file} if train_dataset_file is not None else None
        self.validation_dataset_dict = validation_dataset_file if isinstance(validation_dataset_file, dict) else {"dataset":validation_dataset_file} if validation_dataset_file is not None else None
        self.predict_dataset_dict = predict_dataset_file if isinstance(predict_dataset_file, dict) else {"dataset":predict_dataset_file} if predict_dataset_file is not None else None
        if (
                # only prediction datasets can use 'all' as a label
                # this is because by definition, in training and validation, 'all' refers to the sum of all datasets
                # we can use all in prediction to get out every task regardless of source dataset
                (self.train_dataset_dict and "all" in self.train_dataset_dict)
                or (self.validation_dataset_dict and "all" in self.validation_dataset_dict)
        ):
            raise ValueError("'all' is a reserved keyword and cannot be used as a dataset label.")
        if batch_size!=1:
            warnings.warn(f"Batch size must be 1, you picked {batch_size}. This will be overridden. To get higher effective batch size use accumulate_grad_batches.")
        self.batch_size = 1
        self.transforms = transforms
        self.epoch_size = epoch_size
        self.val_epoch_size = val_epoch_size
        self.dataset_weights = dataset_weights
        self.dataset_class = dataset_class
        self.num_workers = num_workers

    def _create_datasets_from_dict(self, dataset_dict, transforms=None, **kwargs):
        """Create dataset instances from a dict of {key: file_path(s)}."""
        return {
            key: self.dataset_class(files, transforms=transforms or [], **kwargs)
            for key, files in dataset_dict.items()
        }

    def setup(self, stage=None):
        if stage in (None, "fit"):
            if self.train_dataset_dict:
                train_datasets = self._create_datasets_from_dict(
                    self.train_dataset_dict,
                    transforms=self.transforms,
                    batch_size=None,
                )
                self.train_dataset = CompositeDataset(
                    train_datasets,
                    sample_with_replacement=True,
                    epoch_size=self.epoch_size,
                    weights=self.dataset_weights,
                )
            if self.validation_dataset_dict:
                val_datasets = self._create_datasets_from_dict(
                    self.validation_dataset_dict,
                    batch_size=None,
                )
                self.val_dataset = CompositeDataset(
                    val_datasets,
                    sample_with_replacement=False,
                    epoch_size=self.val_epoch_size,
                )
        if stage in (None, "predict"):
            if self.transforms:
                transform_names = []
                for t in self.transforms:
                    if isinstance(t, functools.partial):
                        # Get the actual function/class from the partial
                        transform_names.append(t.func.__name__)
                    else:
                        transform_names.append(type(t).__name__)
                
                warnings.warn(
                    "The following transforms will be applied to the prediction dataset: " + 
                    ", ".join(transform_names) + 
                    ". Make sure this is intended behavior as it may alter predictions."
                )
            if self.predict_dataset_dict:
                predict_datasets = self._create_datasets_from_dict(
                    self.predict_dataset_dict,
                    transforms=self.transforms,
                    batch_size=None,
                    return_specifiers=True,
                )
                self.predict_dataset = CompositeDataset(
                    predict_datasets,
                    sample_with_replacement=False,
                )

    def train_dataloader(self):
        if self.train_dataset_dict is None:
            raise ValueError("Train dataset is not set. Provide `train_dataset_file`.")
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def val_dataloader(self):
        if self.validation_dataset_dict is None:
            raise ValueError("Validation dataset is not set. Provide `validation_dataset_file`.")
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def predict_dataloader(self):
        if self.predict_dataset_dict is None:
            raise ValueError("Prediction dataset is not set. Provide `predict_dataset_file`.")
        return DataLoader(self.predict_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def get_io_mappings_str(self):
        if hasattr(self,"train_dataset"):
            return self.train_dataset.get_io_mappings_str()
        elif hasattr(self,"val_dataset"):
            return self.val_dataset.get_io_mappings_str()
        elif hasattr(self,"predict_dataset"):
            return self.predict_dataset.get_io_mappings_str()
        else:
            return ''