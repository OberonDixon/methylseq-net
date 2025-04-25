import h5py
import os
import numpy as np
from lightning.pytorch.callbacks import BasePredictionWriter, Callback
import torch

class HDF5PredictionWriter(BasePredictionWriter):
    def __init__(self, output_dir, write_interval="batch"):
        super().__init__(write_interval)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.file_handles = {}
        self.pred_counter = 0

    def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx):
        # It appears that all ranks send to rank 0 and write out - but if not, then this logic currently breaks
        rank = trainer.global_rank
        if rank!=0:
            raise ValueError(f"Unexpected rank {rank}. Code in callbacks.py::HDF5PredictionWriter needs to be rewritten if ranks are not getting merged for writing, otherwise values will be missed.")
        path = os.path.join(self.output_dir, f"predictions.h5")

        if path not in self.file_handles:
            self.file_handles[path] = h5py.File(path, "w")
            self.file_handles[path].create_dataset("predictions", shape=(0, *prediction.shape[1:]), maxshape=(None, *prediction.shape[1:]), chunks=True)
            self.file_handles[path].create_dataset("indices", shape=(0,), maxshape=(None,), dtype="i8", chunks=True)

        f = self.file_handles[path]
        n = prediction.shape[0]
        curr_size = f["predictions"].shape[0]

        # Resize datasets
        f["predictions"].resize(curr_size + n, axis=0)
        f["predictions"][curr_size:curr_size + n] = prediction.detach().cpu().numpy()

        f["indices"].resize(curr_size + n, axis=0)
        f["indices"][curr_size:curr_size + n] = np.array(batch_indices)

    def on_predict_end(self, trainer, pl_module):
        self._close_all()

    def __del__(self):
        self._close_all()

    def _close_all(self):
        for f in self.file_handles.values():
            try:
                f.close()
            except Exception:
                pass
        self.file_handles.clear()

class GPUMemoryLogger(Callback):
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated() / 1e6  # MB
            pl_module.log("train/gpu_peak_MB", peak_mem, prog_bar=False)

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated() / 1e6
            pl_module.log("val/gpu_peak_MB", peak_mem, prog_bar=False)

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated() / 1e6
            pl_module.log("test/gpu_peak_MB", peak_mem, prog_bar=False)