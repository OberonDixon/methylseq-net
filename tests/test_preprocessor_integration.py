import os
import tempfile
from pathlib import Path
import warnings

import pytest
import multiprocessing

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
        pipeline.process_samples(subset='all',mode='sequential',max_workers=cores_avail)
        # Check that output files are created
        output_files = list(Path(temp_dir).glob("**/*.h5"))
        assert len(output_files) > 0, "No output files were created by the preprocessing pipeline."