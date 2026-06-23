import os
import sys
import shutil
import tempfile
from pathlib import Path
import warnings

import pytest
import torch
import wandb
from captum.attr import IntegratedGradients

import methylseqnet.train as train
import methylseqnet.predict as predict
from methylseqnet.model import ConditionedSeqNN
from methylseqnet.predict import Predictor
from methylseqnet.hub import DEFAULT_BASE, DEFAULT_VERSION
from test_model import get_config_files_with_names, nuke_gin_config

import gin
import gin.config

@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 1,
    reason="Test requires at least one GPU",
)
@pytest.mark.parametrize("config_file", get_config_files_with_names())
def test_train_predict_integration(config_file, monkeypatch):
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
                attribution_peak_kwargs={"peak_threshold":None},
                attribution_class=IntegratedGradients,
                attribution_baseline_kwargs={"attribution_baselines_per_sample": 1},
                attribution_kwargs = {"n_steps": 1},
            )
        except RuntimeError as e:
            if "used in the graph" in str(e) and "allow_unused" in str(e):
                warnings.warn(f"RuntimeError during predict_locus with attributions: {e}. This may be due to the attribution method not being compatible with the model architecture, and can be ignored for the purposes of this integration test.")

        # CLI integration: exercise methylseqnet-predict for both model sources
        # The CLI writes prediction outputs alongside each input file, so copy the
        # datasets into the temp dir to avoid polluting tests/data. Datasets may be
        # listed per-key as a single path or a list; normalize to one file per key.
        cli_data_dir = Path(temp_dir) / "cli_data"
        cli_data_dir.mkdir(exist_ok=True)
        cli_keys, cli_files = [], []
        for key, value in trainer.datamodule.train_dataset_dict.items():
            src = value[0] if isinstance(value, (list, tuple)) else value
            dst = cli_data_dir / f"{key}_{Path(src).name}"
            shutil.copy(src, dst)
            cli_keys.append(key)
            cli_files.append(str(dst))

        # Empty --supplemental-outputs keeps the test model-agnostic across the
        # parametrized configs (factorized-only reps aren't available everywhere).
        common_argv = [
            "--dataset-keys", *cli_keys,
            "--dataset-files", *cli_files,
            "--supplemental-outputs",
            "--gpus", "1",
            "--num-workers", "1",
        ]

        # local source: load the checkpoint just trained above via the run identifier.
        # In local mode the output label is the model identifier ("test").
        monkeypatch.setattr(sys, "argv", [
            "methylseqnet-predict",
            "--model-source", "local",
            "--model-identifier", "test",
            "--checkpoints-dir", temp_dir,
            *common_argv,
        ])
        predict.main()
        assert list((cli_data_dir / "test").rglob("predictions.h5")), \
            "local CLI run produced no predictions"

        # local source requires --model-identifier; omitting it is a usage error
        monkeypatch.setattr(sys, "argv", [
            "methylseqnet-predict",
            "--model-source", "local",
            "--checkpoints-dir", temp_dir,
            *common_argv,
        ])
        with pytest.raises(SystemExit):
            predict.main()

        # huggingface source: mock the hub download to return a local checkpoint so the
        # CLI from_release plumbing is exercised without any network access.
        fake_ckpt = str(checkpoints_dir / "temp-checkpoint.ckpt")
        monkeypatch.setattr(predict, "release_checkpoint_path", lambda *args, **kwargs: fake_ckpt)

        # --model-identifier is rejected when the source is huggingface
        monkeypatch.setattr(sys, "argv", [
            "methylseqnet-predict",
            "--model-source", "huggingface",
            "--model-identifier", "test",
            *common_argv,
        ])
        with pytest.raises(SystemExit):
            predict.main()

        # valid huggingface invocation; output label is "<base>-<version>"
        monkeypatch.setattr(sys, "argv", [
            "methylseqnet-predict",
            "--model-source", "huggingface",
            *common_argv,
        ])
        predict.main()
        hf_label = f"{DEFAULT_BASE}-{DEFAULT_VERSION}"
        assert list((cli_data_dir / hf_label).rglob("predictions.h5")), \
            "huggingface CLI run produced no predictions"