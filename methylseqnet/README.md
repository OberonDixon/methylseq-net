# Core workflows

## preprocess
`preprocess.py --config /path/to/gin/config.gin` will run a configured preprocessing pipeline that creates .h5 dataset files to be used in training or inference. 

The preprocess pipeline instantiates objects from the `builders` and `writers` modules. `builders` use `readers` to read genomic data files and `encoding` to encode DNA strands.

## train
`train.py --config /path/to/gin/config.gin` will run a training pipeline that creates a trained .ckpt files that can be loaded for future inference.

The training pipeline instantiates a lightning module from `model` and a lightning data module from `dataset` and `datamodule`, with training callbacks from `callbacks` collecting training metrics. `layers`, `losses`, `metrics`, `pretrained`, `tensor_ops`, and `transforms` contain training pipeline submodules.

## inference
`inference.run_dataset_save_h5` runs a dataset through a trained model and writes a predictions h5 file using `callbacks::HDF5PredictionWriter`. 