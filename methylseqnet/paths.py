import tomllib
from pathlib import Path

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"
_CONFIG_PATH = _CONFIG_DIR / "paths.toml"
_EXAMPLE_PATH = _CONFIG_DIR / "paths.toml.example"

if not _CONFIG_PATH.exists():
    raise FileNotFoundError(
        f"Local paths config not found at {_CONFIG_PATH}\n"
        f"Copy the example and edit it for your machine:\n"
        f"  cp {_EXAMPLE_PATH} {_CONFIG_PATH}"
    )

with open(_CONFIG_PATH, "rb") as f:
    _config = tomllib.load(f)

pacbio_5mC_tracks = Path(_config["pacbio_5mC_tracks"])
fiberseq_tracks = Path(_config["fiberseq_tracks"])
rna_tracks = Path(_config["rna_tracks"])
preprocessed_datasets = Path(_config["preprocessed_datasets"])
genomes = Path(_config["genomes"])
model_checkpoints = Path(_config["model_checkpoints"])