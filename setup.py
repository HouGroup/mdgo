# coding: utf-8
# Copyright (c) Tingzheng Hou.
# Distributed under the terms of the MIT License.

import os

from setuptools import setup, find_packages

module_dir = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    setup(
        name='mdgo',
        version='2021.1.15',
        packages=find_packages(),
        install_requires=["tqdm", "pymatgen", "matplotlib", "statsmodels", "re",
                          "MDAnalysis==2.0.0-dev0", "pandas", "numpy", "scipy"],
        description='Repository for analyzing MD results',
        python_requires='>=3.6'
    )
