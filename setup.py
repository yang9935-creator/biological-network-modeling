from setuptools import setup, find_packages

setup(
    name="biocompressor",
    version="0.1.0",
    description="Research toolkit for biological network approximation",
    author="Yuchen Yang",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'scipy>=1.7.0',
        'scikit-learn>=1.0.0',
        'torch>=1.9.0',
        'matplotlib>=3.4.0',
        'seaborn>=0.11.0',
        'plotly>=5.3.0',
        'pyyaml>=5.4.0',
        'tqdm>=4.62.0',
    ],
    extras_require={
        'dev': [
            'jupyter',
            'ipykernel',
            'tensorboard',
            'pytest',
            'pytest-cov',
        ],
        'viz': [
            'kaleido',
            'dash',
        ]
    },
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)
