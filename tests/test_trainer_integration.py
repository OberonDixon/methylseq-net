import os
import tempfile
from pathlib import Path
import warnings

import pytest
import torch
import wandb

import methylseqnet.trainer as trainer
from test_methylseqnn import MethylSeqNN, get_config_files_with_names, nuke_gin_config

import gin
import gin.config

@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 1,
    reason="Test requires at least one GPU",
)
@pytest.mark.parametrize("config_file", get_config_files_with_names())
def test_trainer_integration(config_file):
    nuke_gin_config()
    wandb.finish()
    with tempfile.TemporaryDirectory() as temp_dir:
        os.environ["WANDB_MODE"] = "offline"
        model = trainer.main(
            config=config_file,
            output_dir=temp_dir,
            unique_identifier="test",
            gpus='auto',
            batch_size=1,
            max_epochs=1,
            no_haplotype_metrics=True,
            track_gradients_for_modules=[
                'embeddings_to_methyl_rep',
                'embeddings_to_seq_rep',
                'factorized_reps_to_output',
            ],
        )
        # Check that temp_dir/unique_identifier/checkpoints folder contains a checkpoint for each stage
        checkpoints_dir = Path(f"{temp_dir}/test/checkpoints")
        checkpoint_files = set([f for f in os.listdir(checkpoints_dir) if f.endswith(".ckpt")])
        assert "temp-checkpoint.ckpt" in checkpoint_files
        if model.train_stages:
            for stage_name in model.train_stages.keys():
                assert f"best-checkpoint-{stage_name}.ckpt" in checkpoint_files, f"best-checkpoint-{stage_name}.ckpt not found in {checkpoints_dir}, only found {checkpoint_files}. In the future this will raise an error."
        else:
            assert "best-checkpoint.ckpt" in checkpoint_files
        # Check that checkpoints can be loaded
        for ckpt_file in checkpoint_files:
            model = MethylSeqNN.load_from_checkpoint(checkpoints_dir / ckpt_file)
            assert isinstance(model, MethylSeqNN)
        # Check wandb directory contains a run folder
        wandb_dir = Path(f"{temp_dir}/test/wandb")
        assert any(wandb_dir.iterdir())