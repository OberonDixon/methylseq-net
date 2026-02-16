# Core workflows

## preprocess
`preprocess.py --config /path/to/gin/config.gin` will run a configured preprocessing pipeline that creates .h5 dataset files to be used in training or predict. 

The preprocess pipeline instantiates objects from the `builders` and `writers` modules. `builders` use `readers` to read genomic data files and `encoding` to encode DNA strands.

## train
`train.py --config /path/to/gin/config.gin` will run a training pipeline that creates a trained .ckpt files that can be loaded for future predict.

The training pipeline instantiates a lightning module from `model` and a lightning data module from `dataset` and `datamodule`, with training callbacks from `callbacks` collecting training metrics. `layers`, `losses`, `metrics`, `pretrained`, `tensor_ops`, and `transforms` contain training pipeline submodules.

## predict
`predict.py --model-identifier model_descriptor --dataset-keys key1 key2 --dataset-files /path/to/file1.h5 /path/to/file2.h5` runs datasets through a trained model, using `readers->builders->encoding` to construct tensors, optionally runs `transforms`, and writes a predictions h5 file using `callbacks::HDF5PredictionWriter`. 