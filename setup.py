# coding: utf-8
# Copyright (c) Tingzheng Hou.
# Distributed under the terms of the MIT License.

import os

from setuptools import setup, find_packages

module_dir = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(module_dir, "README.md"), 'r') as f:
    readme = f.read()

INSTALL_REQUIRES = [
    "numpy>=1.16.0",
    "pandas",
    "matplotlib",
    "scipy",
    "tqdm",
    "pymatgen",
    "statsmodels",
    "pubchempy",
    "MDAnalysis",
    "selenium"
]

on_rtd = os.environ.get('READTHEDOCS') == 'True'
if on_rtd:
    INSTALL_REQUIRES = []

if __name__ == "__main__":
    setup(
        name='mdgo',
        version='0.1.0',
        description='A codebase for MD simulation setup and results analysis.',
        long_description=readme,
        long_description_content_type="text/markdown",
        license="MIT",
        author="mdgo development team",
        author_email='tingzheng_hou@berkeley.edu',
        maintainer="Tingzheng Hou",
        maintainer_email='tingzheng_hou@berkeley.edu',
        url="https://github.com/htz1992213/mdgo",
        packages=find_packages(),
        install_requires=INSTALL_REQUIRES,
        extras_require={
            'web': [
                'sphinx',
                'sphinx_rtd_theme',
            ]
        },
        python_requires='>=3.6'
    )

