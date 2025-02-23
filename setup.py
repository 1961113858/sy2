from setuptools import setup, find_packages

setup(
    name="scene_synthesis",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch>=1.7.0",
        "torchvision>=0.8.0",
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
        "hydra-core>=1.1.0",
        "omegaconf>=2.1.0",
    ],
) 