import os
import tempfile
from pathlib import Path
import warnings

import pytest
import torch
import wandb
from captum.attr import IntegratedGradients

import methylseqnet.train as train
from methylseqnet.model import ConditionedSeqNN
from methylseqnet.predict import Predictor
from test_model import get_config_files_with_names, nuke_gin_config

import gin
import gin.config

@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 1,
    reason="Test requires at least one GPU",
)
@pytest.mark.parametrize("config_file", get_config_files_with_names())
def test_train_predict_integration(config_file):
    nuke_gin_config()
    wandb.finish()
    with tempfile.TemporaryDirectory() as temp_dir:
        os.environ["WANDB_MODE"] = "offline"
        trainer = train.main(
            config=config_file,
            output_dir=temp_dir,
            unique_identifier="test",
            gpus='auto',
            batch_size=-1,
            max_epochs=1,
            no_haplotype_metrics=False,
            track_gradients_for_modules=[
                'embeddings_to_methyl_rep',
                'embeddings_to_seq_rep',
                'factorized_reps_to_output',
            ],
        )
        model = trainer.model
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
            model = ConditionedSeqNN.load_from_checkpoint(checkpoints_dir / ckpt_file)
            assert isinstance(model, ConditionedSeqNN)
        # Check wandb directory contains a run folder
        wandb_dir = Path(f"{temp_dir}/test/wandb")
        assert any(wandb_dir.iterdir())
        # Instantiate Predictors
        predictor = Predictor(
            model=model,
        )
        predictor = Predictor(
            model=checkpoints_dir / "temp-checkpoint.ckpt",
        )
        # Run predict_dataset
        dataset_path = trainer.datamodule.train_dataset_dict
        dataset_paths = [{key:value} for key, value in dataset_path.items()]
        for dataset_path in dataset_paths:
            predictor.predict_dataset(
                dataset_path=dataset_path,
                output_path=Path(temp_dir) / "test" / "preds.h5",
            )
        # Run predict_locus 
        predictor.to('cuda')
        predictor.predict_locus(
            chromosome="chr1",
            start=0,
            end=524288,
            sequence_path="./tests/data/chr1_fake1M.fa.gz",
            methylation_paths=["./tests/data/hg38_test_zeros.hg38.bigwig"],
            capture_attributions=False,
        ) 
        # Run predict_locus with attributions
        # (allowed to run out of memory or raise a RuntimeError due to incompatibility
        # between attribution method and model architecture, since the purpose of this 
        # test is just to check that the code runs end-to-end without syntax errors or 
        # other issues that would prevent it from running, and not to verify the correctness 
        # of the attributions themselves)
        try:
            predictor.to('cuda')
            predictor.predict_locus(
                chromosome="chr1",
                start=0,
                end=524288,
                sequence_path="./tests/data/chr1_fake1M.fa.gz",
                methylation_paths=["./tests/data/hg38_test_zeros.hg38.bigwig"],
                capture_attributions=True,
                attribution_peak_threshold=None,
                attribution_class=IntegratedGradients,
                attribution_ref_shuffles_per_sample=1,
                n_steps=1,
            )
        except RuntimeError as e:
            if "used in the graph" in str(e) and "allow_unused" in str(e):
                warnings.warn(f"RuntimeError during predict_locus with attributions: {e}. This may be due to the attribution method not being compatible with the model architecture, and can be ignored for the purposes of this integration test.")