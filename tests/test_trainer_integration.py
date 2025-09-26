import os
import tempfile
from pathlib import Path
import warnings

import pytest
import torch
import wandb

import methylseqnet.trainer as trainer
from methylseqnet.methylseqnn import MethylSeqNN

import gin
import gin.config

def nuke_gin_config():
    gin.clear_config()
    # Clear operative config
    if hasattr(gin.config, 'clear_operative_config'):
        gin.config.clear_operative_config()
    
    # Clear all internal registries
    if hasattr(gin.config, '_CONFIGURABLE_SINGLETONS'):
        gin.config._CONFIGURABLE_SINGLETONS.clear()
    if hasattr(gin.config, '_CONFIGURABLE_FUNCTIONS'):
        gin.config._CONFIGURABLE_FUNCTIONS.clear()
    if hasattr(gin.config, '_CONFIGURABLE_CLASSES'):
        gin.config._CONFIGURABLE_CLASSES.clear()
    if hasattr(gin.config, '_CONSTANTS'):
        gin.config._CONSTANTS.clear()
    
    # Reset config state
    if hasattr(gin.config, '_OPERATIVE_CONFIG_STR'):
        gin.config._OPERATIVE_CONFIG_STR = ""
    if hasattr(gin.config, '_CONFIG_IS_LOCKED'):
        gin.config._CONFIG_IS_LOCKED = False

def get_config_files():
    """Find all gin config files relative to the test file location."""
    test_file_dir = Path(__file__).parent
    configs_dir = test_file_dir / "configs"
    
    # If configs not found next to test file, try looking in repo root
    if not configs_dir.exists():
        # Go up directories until we find the repo root (contains setup.py, pyproject.toml, etc.)
        current_dir = test_file_dir
        while current_dir.parent != current_dir:  # Stop at filesystem root
            if any((current_dir / marker).exists() for marker in ["setup.py", "pyproject.toml", ".git"]):
                configs_dir = current_dir / "tests" / "configs"
                break
            current_dir = current_dir.parent
    
    if not configs_dir.exists():
        pytest.skip(f"Could not find configs directory. Looked in {configs_dir}")
    
    # Get all .gin files
    config_files = list(configs_dir.glob("*.gin"))
    # config_files = [config_files_list[4]]
    if not config_files:
        pytest.skip(f"No .gin files found in {configs_dir}")
    
    return config_files

def get_config_files_with_names():
    """Get config files with readable test names."""
    config_files = get_config_files()
    return [
        pytest.param(config_file, id=config_file.stem)
        for config_file in config_files
    ]

@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 1,
    reason="Test requires at least one GPU",
)
@pytest.mark.parametrize("config_file", get_config_files_with_names())
def test_trainer_integration(config_file):
    nuke_gin_config()
    with tempfile.TemporaryDirectory() as temp_dir:
        os.environ["WANDB_MODE"] = "offline"
        model = trainer.main(
            config=config_file,
            output_dir=temp_dir,
            unique_identifier="test",
            gpus='auto',
            batch_size=1,
            no_haplotype_metrics=True,
        )
        # Check that temp_dir/unique_identifier/checkpoints folder contains a checkpoint for each stage
        checkpoints_dir = Path(f"{temp_dir}/test/checkpoints")
        checkpoint_files = set([f for f in os.listdir(checkpoints_dir) if f.endswith(".ckpt")])
        assert "temp-checkpoint.ckpt" in checkpoint_files
        if model.train_stages:
            for stage_name in model.train_stages.keys():
                # warning if not present
                if f"best-checkpoint-{stage_name}.ckpt" not in checkpoint_files:
                    warnings.warn(f"Warning: best-checkpoint-{stage_name}.ckpt not found in {checkpoints_dir}, only found {checkpoint_files}. In the future this will raise an error.")
        else:
            assert "best-checkpoint.ckpt" in checkpoint_files
        # Check that checkpoints can be loaded
        for ckpt_file in checkpoint_files:
            model = MethylSeqNN.load_from_checkpoint(checkpoints_dir / ckpt_file)
            assert isinstance(model, MethylSeqNN)
        # Check wandb directory contains a run folder
        wandb_dir = Path(f"{temp_dir}/test/wandb")
        assert any(wandb_dir.iterdir())

        wandb.finish()
        del os.environ["WANDB_MODE"]