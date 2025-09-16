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
        # Initialize the callback
        io_mappings_str = "input_key->output_key,another_input->another_output"
        writer = HDF5PredictionWriter(
            output_dir=temp_dir,
            write_interval="batch",
            io_mappings_str=io_mappings_str
        )
        
        # Create mock trainer and pl_module
        mock_trainer = Mock()
        mock_trainer.global_rank = 0  # Simulate rank 0
        mock_pl_module = Mock()
        mock_pl_module.trim_targets = lambda inputs,targets: targets  # Identity function for trimming
        
        pred_shape = (10, 3)  # Example prediction shape
        
        # Batch 1 data
        predictions_1 = torch.randn(4, *pred_shape)
        specifiers_1 = [f"sample_{i}" for i in range(4)]
        batch_indices_1 = [0, 1, 2, 3]
        
        prediction_1 = {
            "predictions": predictions_1,
            "specifiers": specifiers_1
        }

        mock_batch_1 = (Mock(), torch.randn(4, *pred_shape), Mock(), Mock())
        
        # Batch 2 data (simulate non-contiguous indices)
        predictions_2 = torch.randn(3, *pred_shape)
        specifiers_2 = [f"sample_{i}" for i in range(4, 7)]
        batch_indices_2 = [4, 5, 6]
        
        prediction_2 = {
            "predictions": predictions_2,
            "specifiers": specifiers_2
        }

        mock_batch_2 = (Mock(), torch.randn(3, *pred_shape), Mock(), Mock())
        
        # Test writing first batch
        writer.write_on_batch_end(
            trainer=mock_trainer,
            pl_module=mock_pl_module,
            prediction=prediction_1,
            batch_indices=batch_indices_1,
            batch=mock_batch_1,
            batch_idx=0,
            dataloader_idx=0
        )
        
        # Test writing second batch
        writer.write_on_batch_end(
            trainer=mock_trainer,
            pl_module=mock_pl_module,
            prediction=prediction_2,
            batch_indices=batch_indices_2,
            batch=mock_batch_2,
            batch_idx=1,
            dataloader_idx=0
        )
        
        # Test prediction end (closes files)
        writer.on_predict_end(mock_trainer, mock_pl_module)
        
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
        
        # Test error case: non-zero rank should raise ValueError
        mock_trainer.global_rank = 1
        
        with pytest.raises(ValueError, match="Unexpected rank 1"):
            writer.write_on_batch_end(
                trainer=mock_trainer,
                pl_module=mock_pl_module,
                prediction=prediction_1,
                batch_indices=batch_indices_1,
                batch=mock_batch_1,
                batch_idx=0,
                dataloader_idx=0
            )


def test_hdf5_prediction_writer_file_cleanup():
    """Test that files are properly closed and cleaned up."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        writer = HDF5PredictionWriter(output_dir=temp_dir)
        
        # Create mock data
        mock_trainer = Mock()
        mock_trainer.global_rank = 0
        mock_pl_module = Mock()
        mock_pl_module.trim_targets = lambda inputs,targets: targets  # Identity function for trimming
        
        predictions = torch.randn(2, 5, 3)
        pred_shape = predictions.shape[1:]
        specifiers = ["sample_0", "sample_1"]
        batch_indices = [0, 1]
        
        prediction = {
            "predictions": predictions,
            "specifiers": specifiers
        }

        mock_batch = (Mock(), torch.randn(2, *pred_shape), Mock(), Mock())
        
        # Write some data to create file handles
        writer.write_on_batch_end(
            trainer=mock_trainer,
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