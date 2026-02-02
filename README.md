# methylseq-net

MethylSeqNet is a method for conditioning genomic regulatory activity predictons on epigenetic state, currently in the form of CpG methylation landscape. This repo was created based on preliminary work in April 2023. 

## Installation

Clone the repository and navigate into the top-level directory containing `environment.yml` and `pyproject.toml`. 

```
git clone https://github.com/OberonDixon/methylseq-net
cd methylseq-net
```

Create a conda environment by running this following command. This will install the `methylseqnet` python package in editable mode so any code changes will be reflected whenever the python kernel is re-started. Conda dependencies necessary for preprocessing and downstream analysis are also installed.

```
conda env create -f environment.yml
```

You can also update your existing environment:

```
conda env update -f environment.yml
```

And you can update while removing any unnecessary dependencies:

```
conda env update -f environment.yml --prune
```