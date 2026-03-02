from pathlib import Path
import pytest

from methylseqnet.model import ConditionedSeqNN

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

def get_config_files(config_subdir: str = "configs"):
    """Find all gin config files relative to the test file location."""
    test_file_dir = Path(__file__).parent
    configs_dir = test_file_dir / config_subdir
    
    # If configs not found next to test file, try looking in repo root
    if not configs_dir.exists():
        # Go up directories until we find the repo root (contains setup.py, pyproject.toml, etc.)
        current_dir = test_file_dir
        while current_dir.parent != current_dir:  # Stop at filesystem root
            if any((current_dir / marker).exists() for marker in ["setup.py", "pyproject.toml", ".git"]):
                configs_dir = current_dir / "tests" / config_subdir
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

def get_config_files_with_names(config_subdir: str = "configs"):
    """Get config files with readable test names."""
    config_files = get_config_files(config_subdir)
    return [
        pytest.param(config_file, id=config_file.stem)
        for config_file in config_files
    ]

@pytest.mark.parametrize("config_file", get_config_files_with_names())
def test_model_initialization(config_file):
    nuke_gin_config()
    gin.parse_config_file(config_file)
    model = ConditionedSeqNN()
    assert isinstance(model, ConditionedSeqNN)