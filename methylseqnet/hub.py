from typing import Literal
import logging
from pathlib import Path

from huggingface_hub import hf_hub_download, try_to_load_from_cache

logger = logging.getLogger(__name__)

DEFAULT_REPO = "OberonDixon/methylseqnet"
DEFAULT_BASE = "borzoi-rep0"
DEFAULT_VERSION = "v1.0"

def release_checkpoint_path(
    base: Literal["borzoi-rep0"] = DEFAULT_BASE,
    version: str = DEFAULT_VERSION,
    *,
    repo_id: str = DEFAULT_REPO,
    filename: str | None = None,
):
    filename = f"factorized-{base}.ckpt" if filename is None else filename
    
    cached = try_to_load_from_cache(repo_id, filename, revision=version)  # no network
    if isinstance(cached, str):
        ckpt_path = cached
        method = "Cached"
    else:  # None (not cached) or _CACHED_NO_EXIST sentinel -> go fetch
        ckpt_path = hf_hub_download(repo_id=repo_id, filename=filename, revision=version)
        method = "Downloaded"

    sha = Path(ckpt_path).parent.name
    
    logger.info("%s %s from %s @ %s (%s)", method, filename, repo_id, version, sha)

    return ckpt_path