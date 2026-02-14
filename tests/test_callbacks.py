import pytest
import tempfile
import shutil
import os
import h5py
import numpy as np
import torch
from unittest.mock import Mock, MagicMock

from methylseqnet.callbacks import (
    HDF5PredictionWriter,
    GPUMemoryLogger,
    HaplotypedPredLogger,
)

def test_hdf5_prediction_writer():
    """Test HDF5PredictionWriter callback with dummy data and temporary directory."""
    
    # Create temporary directory that auto-cleans
    with tempfile.TemporaryDirectory() as temp_dir:
        io_mappings_str = "mapping"
        # Initialize the callback
        writer = HDF5PredictionWriter(
            output_dir=temp_dir,
            write_interval="batch",
        )
        
        # Create mock train and pl_module
        mock_train = Mock()
        mock_train.global_rank = 0  # Simulate rank 0
        mock_train.world_size = 1
        mock_pl_module = Mock()
        mock_pl_module.crop_targets = lambda targets: targets  # Identity function for trimming
        mock_pl_module.io_mappings_str = io_mappings_str
        
        pred_shape = (1, 10, 3)  # Example prediction shape
        
        # Batch 1 data
        predictions_1 = torch.randn(4, *pred_shape)
        specifiers_1 = [f"sample_{i}" for i in range(4)]
        batch_indices_1 = [0, 1, 2, 3]
        
        prediction_1 = {
            "predictions": predictions_1,
            "specifier": specifiers_1
        }

        mock_batch_1 = {
            'sequence':Mock(),
            'target':torch.randn(4, *pred_shape),
            'mask':Mock(),
            'specifier':Mock(),
        }
        
        # Batch 2 data (simulate non-contiguous indices)
        predictions_2 = torch.randn(3, *pred_shape)
        specifiers_2 = [f"sample_{i}" for i in range(4, 7)]
        batch_indices_2 = [4, 5, 6]
        
        prediction_2 = {
            "predictions": predictions_2,
            "specifier": specifiers_2
        }

        mock_batch_2 = {
            'sequence':Mock(),
            'target':torch.randn(3, *pred_shape),
            'mask':Mock(),
            'specifier':Mock(),
        }
        
        # Test writing first batch
        writer.write_on_batch_end(
            train=mock_train,
            pl_module=mock_pl_module,
            prediction=prediction_1,
            batch_indices=batch_indices_1,
            batch=mock_batch_1,
            batch_idx=0,
            dataloader_idx=0
        )
        
        # Test writing second batch
        writer.write_on_batch_end(
            train=mock_train,
            pl_module=mock_pl_module,
            prediction=prediction_2,
            batch_indices=batch_indices_2,
            batch=mock_batch_2,
            batch_idx=1,
            dataloader_idx=0
        )
        
        # Test prediction end (closes files)
        writer.on_predict_end(mock_train, mock_pl_module)
        
        # Verify the HDF5 file was created and contains expected data
        h5_path = os.path.join(temp_dir, "predictions.h5")
        assert os.path.exists(h5_path), "HDF5 file should be created"
        
        # Read and verify the data
        with h5py.File(h5_path, "r") as f:
            # Check datasets exist
            assert "predictions" in f, "Predictions dataset should exist"
            assert "indices" in f, "Indices dataset should exist"
            assert "specifier" in f, "Specifier dataset should exist"
            
            # Check io_mappings attribute
            assert f.attrs['io_mappings'] == io_mappings_str, "IO mappings should be stored as attribute"
            
            # Check data shapes and content
            stored_predictions = f["predictions"][:]
            stored_indices = f["indices"][:]
            stored_specifiers = f["specifier"][:]
            
            # Should have 7 samples total (indices 0-6)
            assert stored_predictions.shape[0] == 7, "Should have 7 prediction samples"
            assert stored_indices.shape[0] == 7, "Should have 7 index samples"
            assert stored_specifiers.shape[0] == 7, "Should have 7 specifier samples"
            
            # Check prediction shape matches
            assert stored_predictions.shape[1:] == pred_shape, f"Prediction shape should be {pred_shape}"
            
            # Verify indices are correct
            expected_indices = np.array([0, 1, 2, 3, 4, 5, 6])
            np.testing.assert_array_equal(stored_indices[:7], expected_indices)
            
            # Verify some predictions match (first batch)
            np.testing.assert_array_almost_equal(
                stored_predictions[0], 
                predictions_1[0].detach().cpu().numpy(),
                decimal=5
            )
            
            # Verify specifiers
            expected_specifiers = [f"sample_{i}".encode('utf-8') for i in range(7)]
            assert list(stored_specifiers[:7]) == expected_specifiers


def test_hdf5_prediction_writer_file_cleanup():
    """Test that files are properly closed and cleaned up."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        writer = HDF5PredictionWriter(output_dir=temp_dir)
        
        # Create mock data
        mock_train = Mock()
        mock_train.global_rank = 0
        mock_train.world_size = 1
        mock_pl_module = Mock()
        mock_pl_module.crop_targets = lambda targets: targets  # Identity function for trimming
        mock_pl_module.io_mappings_str = ""
        
        predictions = torch.randn(2, 5, 3)
        pred_shape = predictions.shape[1:]
        specifiers = ["sample_0", "sample_1"]
        batch_indices = [0, 1]
        
        prediction = {
            "predictions": predictions,
            "specifier": specifiers
        }

        mock_batch = {
            'sequence':Mock(),
            'target':torch.randn(2, *pred_shape),
            'mask':Mock(),
            'specifier':Mock(),
        }
        
        # Write some data to create file handles
        writer.write_on_batch_end(
            train=mock_train,
            pl_module=mock_pl_module,
            prediction=prediction,
            batch_indices=batch_indices,
            batch=mock_batch,
            batch_idx=0,
            dataloader_idx=0
        )
        
        # Verify file handle exists
        assert len(writer.file_handles) == 1, "Should have one file handle"
        
        # Test _close_all method
        writer._close_all()
        assert len(writer.file_handles) == 0, "File handles should be cleared"
        
        # Test that __del__ doesn't crash (simulate garbage collection)
        writer.__del__()