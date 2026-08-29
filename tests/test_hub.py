from unittest import mock

import pytest
import torch

import methylseqnet.hub as hub
from methylseqnet.hub import (
    release_checkpoint_path,
    DEFAULT_REPO,
    DEFAULT_BASE,
    DEFAULT_VERSION,
)

DEFAULT_FILENAME = f"factorized-{DEFAULT_BASE}.ckpt"


def test_release_checkpoint_path_uses_cache_when_available():
    """A cached file (str path) is returned directly without downloading."""
    cached = "/fake/cache/models--repo/snapshots/abc123/" + DEFAULT_FILENAME
    with mock.patch.object(hub, "try_to_load_from_cache", return_value=cached) as mock_cache, \
         mock.patch.object(hub, "hf_hub_download") as mock_download:
        path = release_checkpoint_path()

    assert path == cached
    mock_cache.assert_called_once_with(DEFAULT_REPO, DEFAULT_FILENAME, revision=DEFAULT_VERSION)
    mock_download.assert_not_called()


def test_release_checkpoint_path_downloads_when_not_cached():
    """None from the cache lookup means not cached, so we download."""
    downloaded = "/fake/hub/models--repo/snapshots/def456/" + DEFAULT_FILENAME
    with mock.patch.object(hub, "try_to_load_from_cache", return_value=None), \
         mock.patch.object(hub, "hf_hub_download", return_value=downloaded) as mock_download:
        path = release_checkpoint_path()

    assert path == downloaded
    mock_download.assert_called_once_with(
        repo_id=DEFAULT_REPO, filename=DEFAULT_FILENAME, revision=DEFAULT_VERSION
    )


def test_release_checkpoint_path_downloads_on_cache_no_exist_sentinel():
    """A non-str sentinel (_CACHED_NO_EXIST) should download, not be used as a path."""
    sentinel = object()  # stand-in for huggingface_hub's _CACHED_NO_EXIST
    downloaded = "/fake/hub/x/" + DEFAULT_FILENAME
    with mock.patch.object(hub, "try_to_load_from_cache", return_value=sentinel), \
         mock.patch.object(hub, "hf_hub_download", return_value=downloaded) as mock_download:
        path = release_checkpoint_path()

    assert path == downloaded
    mock_download.assert_called_once()


def test_release_checkpoint_path_default_filename_derived_from_base():
    """With no filename given, it is built as factorized-<base>.ckpt."""
    with mock.patch.object(hub, "try_to_load_from_cache", return_value=None) as mock_cache, \
         mock.patch.object(hub, "hf_hub_download", return_value="/x.ckpt") as mock_download:
        release_checkpoint_path(base="borzoi-rep0")

    # try_to_load_from_cache is called positionally as (repo_id, filename)
    assert mock_cache.call_args.args[1] == "factorized-borzoi-rep0.ckpt"
    assert mock_download.call_args.kwargs["filename"] == "factorized-borzoi-rep0.ckpt"


def test_release_checkpoint_path_passes_through_custom_args():
    """Explicit base/version/repo_id/filename are forwarded to the hub calls."""
    with mock.patch.object(hub, "try_to_load_from_cache", return_value=None) as mock_cache, \
         mock.patch.object(hub, "hf_hub_download", return_value="/custom.ckpt") as mock_download:
        path = release_checkpoint_path(
            "borzoi-rep0", "v9.9", repo_id="someone/else", filename="custom.ckpt"
        )

    assert path == "/custom.ckpt"
    mock_cache.assert_called_once_with("someone/else", "custom.ckpt", revision="v9.9")
    mock_download.assert_called_once_with(
        repo_id="someone/else", filename="custom.ckpt", revision="v9.9"
    )


# Real released checkpoint: fetched via the hub.py DEFAULTs so that bumping
# DEFAULT_VERSION (and publishing a matching checkpoint) keeps these tests pointed
# at the current release without edits here. Borzoi-style models take a fixed
# 524288 bp input window; reuse the fake genome / zeroed methylation track.
RELEASE_INPUT_LENGTH = 524288
FAKE_GENOME = "./tests/data/chr1_fake1M.fa.gz"
ZEROED_METHYL = "./tests/data/hg38_test_zeros.hg38.bigwig"


def _network_errors():
    """Exceptions meaning 'couldn't reach the Hub' rather than a real compat break."""
    errs = [OSError]  # ConnectionError, timeouts, etc. subclass OSError
    try:
        from requests.exceptions import RequestException
        errs.append(RequestException)
    except Exception:
        pass
    try:
        from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError
        errs.extend([HfHubHTTPError, LocalEntryNotFoundError])
    except Exception:
        pass
    return tuple(errs)


NETWORK_ERRORS = _network_errors()


@pytest.fixture(scope="module")
def released_checkpoint_path():
    """Current released checkpoint (DEFAULT base/version), or skip if the Hub is unreachable."""
    try:
        return release_checkpoint_path()
    except NETWORK_ERRORS as e:
        pytest.skip(f"Could not fetch released checkpoint from the Hub (offline?): {e}")


def test_model_from_release_returns_conditioned_seqnn(released_checkpoint_path):
    """ConditionedSeqNN.from_release loads the published checkpoint into a model."""
    from methylseqnet.model import ConditionedSeqNN

    model = ConditionedSeqNN.from_release()
    assert isinstance(model, ConditionedSeqNN)


def test_model_from_pretrained_is_alias_for_from_release(released_checkpoint_path):
    """from_pretrained is a thin alias of from_release and yields the same type."""
    from methylseqnet.model import ConditionedSeqNN

    model = ConditionedSeqNN.from_pretrained()
    assert isinstance(model, ConditionedSeqNN)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 1,
    reason="Forward pass over the full input window requires a GPU",
)
def test_released_checkpoint_runs_forward_pass(released_checkpoint_path):
    """Load the published checkpoint and run a forward pass, guarding against future code changes silently breaking checkpoint compatibility."""
    from methylseqnet.predict import Predictor

    # Uses the cache populated by the released_checkpoint_path fixture
    predictor = Predictor.from_release()

    prediction = predictor.predict_locus(
        chromosome="chr1",
        start=0,
        end=RELEASE_INPUT_LENGTH,
        sequence_path=FAKE_GENOME,
        methylation_paths=[ZEROED_METHYL],
        capture_attributions=False,
    )

    preds = prediction["predictions"]
    assert isinstance(preds, torch.Tensor)
    assert preds.ndim >= 2, f"expected a multi-dim prediction tensor, got shape {preds.shape}"
    # Output is binned at the model stride; length must match the per-bin coordinates
    assert preds.shape[-1] > 0
    assert preds.shape[-1] == prediction["output_coordinates"].shape[-1], \
        "prediction length does not match the number of output bins"
    assert torch.isfinite(preds).all(), "released checkpoint produced non-finite predictions"
