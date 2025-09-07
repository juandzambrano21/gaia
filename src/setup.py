"""
Setup script for the GAIA package.
"""

from setuptools import setup, find_packages

setup(
    name="gaia",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        # Core PyTorch
        "torch>=2.0.0,<2.2.0",
        "torchvision>=0.15.0",
        "torchaudio>=2.0.0",
        
        # Scientific computing
        "numpy<2.0",
        "scipy>=1.10.0",
        
        # Machine learning utilities
        "scikit-learn>=1.3.0",
        
        # Data processing
        "pandas>=2.0.0",
        "datasets>=2.14.0",
        
        # Progress bars and utilities
        "tqdm>=4.65.0",
        
        # Configuration management
        "pyyaml>=6.0",
        
        # Advanced metrics
        "torchmetrics>=1.0.0",
        
        # Natural language processing
        "nltk>=3.8.0",
        
        # Graph processing
        "networkx>=3.1",
        
        # Serialization
        "joblib>=1.3.0",
        
        # Web requests
        "requests>=2.31.0",
        
        # Environment management
        "python-dotenv>=1.0.0",
        
        # Performance monitoring
        "psutil>=5.9.0",
    ],
    extras_require={
        "dev": [
            # Testing
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            
            # Development tools
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
        "optional": [
            # Experiment tracking
            "wandb>=0.15.0",
            
            # Visualization
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            
            # Advanced optimization
            "optuna>=3.3.0",
            
            # CLI utilities
            "click>=8.1.0",
            "rich>=13.0.0",
            
            # Jupyter notebooks
            "jupyter>=1.0.0",
            "ipykernel>=6.25.0",
            
            # Model serving
            "fastapi>=0.100.0",
            "uvicorn>=0.23.0",
            
            # Data validation
            "pydantic>=2.0.0",
            
            # Async support
            "aiofiles>=23.2.0",
            
            # Image processing
            "Pillow>=10.0.0",
        ]
    },
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