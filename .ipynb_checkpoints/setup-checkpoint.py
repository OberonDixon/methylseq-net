from setuptools import find_packages, setup

setup(
    name="methylseqnet",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "torch==2.0.0",
        "biopython",
        "scikit-learn",
        "captum",
        "modisco",
        "scipy",
    ],
)