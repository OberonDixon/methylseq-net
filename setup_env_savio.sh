#!/bin/bash

# Set variables
env_name="methylseqnet"                 # Name of the conda environment
env_dir="/clusterfs/nilah/oberon/environments/"                  # Directory to create the environment in
home_conda_dir="/global/home/users/dixonluinenburg/.conda/envs/"
channels="conda-forge nanoporetech defaults"                 # Channels to install packages from
packages="python=3.11 nanoporetech::modkit==0.2.4 bioconda::samtools"                 # Core packages to install
pip_editable_packages="-e /clusterfs/nilah/oberon/repos/dimelo_v2"        # Local directory containing the pip package

module load python

# Ensure the environment directory exists
mkdir -p $env_dir

# Create the conda environment in the specified directory
conda create --prefix $env_dir/$env_name $packages $(printf -- "--channel %s " $channels) -y

# Activate the environment
source activate $env_dir/$env_name

# Make a symlink in the user's home conda directory
ln -s $env_dir/$env_name $home_conda_dir/$env_name

# Pip install dependencies
pip install $pip_editable_packages

# Install package itself
pip install -e .

# Deactivate the environment
conda deactivate

echo "Environment setup and package installation complete!"