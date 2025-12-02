from setuptools import find_packages, setup

setup(
    name="methylseqnet",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",#
        # "nvidia-cudnn-cu12",#==9.1.0.70
        "biopython",
        "scikit-learn",
        "captum",
        "modisco-lite",
        "scipy",
        "gin-config",
        "pynvml",
        "logomaker",
        "pysam",
        "pytest",
        "lightning", 
        "h5py",
        "pyBigWig",
        "wandb>=0.21",
        "borzoi-pytorch",
        "umap-learn",
        "numpy<2.0.0",
    ],
)