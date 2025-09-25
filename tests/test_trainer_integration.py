import os
import tempfile
from pathlib import Path

import pytest
import torch

import methylseqnet.trainer as trainer
from methylseqnet.methylseqnn import MethylSeqNN

@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 1,
    reason="Test requires at least one GPU",
)
def test_trainer_integration():
    with tempfile.TemporaryDirectory() as temp_dir:
        os.environ["WANDB_MODE"] = "offline"
        os.environ["WANDB_DIR"] = temp_dir
        model = trainer.main(
            config="./tests/configs/borzoi_probe_test.gin",
            output_dir=temp_dir,
            unique_identifier="test",
            gpus='auto',
            batch_size=1,
        )
        # Check that temp_dir/unique_identifier/checkpoints folder contains a checkpoint for each stage
        checkpoints_dir = Path(f"{temp_dir}/test/checkpoints")
        checkpoint_files = set([f for f in os.listdir(checkpoints_dir) if f.endswith(".ckpt")])
        assert "temp-checkpoint.ckpt" in checkpoint_files
        if model.train_stages:
            for stage_name in model.train_stages.keys():
                assert f"best-checkpoint-{stage_name}.ckpt" in checkpoint_files
        else:
            assert "best-checkpoint.ckpt" in checkpoint_files
        # Check that checkpoints can be loaded
        for ckpt_file in checkpoint_files:
            model = MethylSeqNN.load_from_checkpoint(checkpoints_dir / ckpt_file)
            assert isinstance(model, MethylSeqNN)
        # # Check wandb directory contains a run folder
        # wandb_dir = Path(f"{temp_dir}/wandb")
        # assert any(wandb_dir.iterdir())
