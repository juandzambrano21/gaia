"""
Setup script for the GAIA package.
"""

from setuptools import setup, find_packages

setup(
    name="gaia",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.19.0",
        "tensorboard>=2.5.0",
    ],
    author="GAIA Team",
    author_email="gaia@example.com",
    description="Generative Algebraic Intelligence Architecture",
    keywords="machine learning, algebraic topology, simplicial sets, kan fibrations",
    url="https://github.com/gaia-team/gaia",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)