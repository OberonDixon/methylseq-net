import os
import tempfile
from pathlib import Path
import warnings

import pytest
import multiprocessing
import h5py

import methylseqnet.preprocessor as preprocessor
from test_methylseqnn import MethylSeqNN, get_config_files_with_names, nuke_gin_config

import gin
import gin.config

@pytest.mark.parametrize("config_file", get_config_files_with_names("preprocessor_configs"))
def test_trainer_integration(config_file):
    nuke_gin_config()
    with tempfile.TemporaryDirectory() as temp_dir:
        gin.parse_config_file(config_file)
        pipeline = preprocessor.PreprocessingPipeline(output_directory=temp_dir)
        cores_avail = multiprocessing.cpu_count()
        pipeline.process_samples(subset='all',sequential=True,max_workers=cores_avail)
        # Check that output files are created
        output_files = list(Path(temp_dir).glob("**/*.h5"))
        assert len(output_files) > 0, "No output files were created by the preprocessing pipeline."
        for output_file in output_files:
            with h5py.File(output_file, 'r') as h5f:
                assert 'sequence' in h5f, f"'sequence' dataset not found in {output_file}"
                assert 'methylation' in h5f, f"'methylation' dataset not found in {output_file}"
                assert 'tracks' in h5f, f"'tracks' dataset not found in {output_file}"
                assert len(h5f['sequence']) == len(h5f['tracks']), f"Mismatch in number of sequences and tracks in {output_file}"
                expected_seq_length = pipeline.dataset_writer.seq_length
                expected_track_length = pipeline.dataset_writer.track_length
                expected_num_tracks = pipeline.dataset_writer.num_tracks
                expected_num_cell_types = pipeline.dataset_writer.num_cell_types
                expected_num_variants = pipeline.dataset_writer.num_variants
                assert h5f['sequence'].shape[3] == expected_seq_length, f"Sequence length mismatch in {output_file}"
                assert h5f['sequence'].shape[2] == 4, f"Sequence one-hot encoding dimension mismatch in {output_file}"
                assert h5f['sequence'].shape[1] == expected_num_variants, f"Number of sequence variants mismatch in {output_file}"
                assert h5f['methylation'].shape[4] == expected_seq_length, f"Methylation length mismatch in {output_file}"
                assert h5f['methylation'].shape[3] == 3, f"Methylation channels dimension mismatch in {output_file}"
                assert h5f['methylation'].shape[2] == expected_num_cell_types, f"Methylation cell types dimension mismatch in {output_file}"
                assert h5f['methylation'].shape[1] == expected_num_variants, f"Number of methylation variants mismatch in {output_file}"
                assert h5f['tracks'].shape[3] == expected_track_length, f"Tracks length mismatch in {output_file}"
                assert h5f['tracks'].shape[2] == expected_num_tracks, f"Tracks channels dimension mismatch in {output_file}"
                assert h5f['tracks'].shape[1] == expected_num_variants, f"Number of tracks variants mismatch in {output_file}"
