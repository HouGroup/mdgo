# ![MDGO](https://github.com/HT-MD/mdgo/blob/main/docs/logo_mdgo.svg)

![PyPI - Downloads](https://img.shields.io/pypi/dm/mdgo?style=plastic)

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/a97ce4eb53404e58b89bf41e0c3a3ee6)](https://app.codacy.com/gh/HT-MD/mdgo?utm_source=github.com&utm_medium=referral&utm_content=HT-MD/mdgo&utm_campaign=Badge_Grade_Settings)

[![docs](https://readthedocs.org/projects/mdgo/badge/?version=latest&style=plastic)](https://mdgo.readthedocs.io/) ![Linting](https://github.com/HT-MD/mdgo/actions/workflows/lint.yml/badge.svg) ![Test](https://github.com/HT-MD/mdgo/actions/workflows/test.yml/badge.svg)

An all-in-one code base for the classical molecualr dynamics (MD) simulation setup and results analysis. 

# 1. Installation

## 1.1 Installing from PyPI

To install the latest release version of mdgo:

`pip install mdgo`
    
## 1.2 Installing from source code

Mdgo requires numpy, pandas, matplotlib, scipy, tqdm, statsmodels, pymatgen>=2022.0.9, pubchempy, selenium, MDAnalysis (version 2.0.0-dev0 prefered) and their dependencies.           

### Getting Source Code

If not available already, use the following steps.

1. Install [git](http://git-scm.com), if not already packaged with your system.

2. Download the mdgo source code using the command:

   `git clone https://github.com/htz1992213/mdgo.git`
    
### Installation

1. Navigate to mdgo root directory:

   `cd mdgo`

2. Install the code, using the command:

   `pip install .`

3. The latest version MDAnalysis==2.0.0.dev0 is recommended, you may download the source code of the latest MDAnalysis from github and install using pip to replace an existing version.

### Installation in development mode

1. Navigate to mdgo root directory:

   `cd mdgo`

2. Install the code in "editable" mode, using the command::

   `pip install -e .`

3. The latest version MDAnalysis==2.0.0.dev0 is recommended, you may download the source code of the latest MDAnalysis from github and install using pip to replace an existing version.

## 2. Features

1.  Retrieving compound structure and information from PubChem
    -   Supported searching text:
        -   cid, name, smiles, inchi, inchikey or formula
    -   Supported output format:
        -   smiles code, PDB, XML, ASNT/B, JSON, SDF, CSV, PNG, TXT
2.  Retrieving water and ion models
    -   Supported water models:
        -   SCP, SPC/E, TIP3P_EW, TIP4P_EW, TIP4P_2005
    -   Supported ion models:
        -   alkali, ammonium, and halide monovalent ions by Jensen and Jorgensen 
        -   alkali and halide monovalent ions by Joung and Cheatham
        -   alkali and alkaline-earth metal cations by Åqvist
3.  Write OPLS-AA forcefield file from LigParGen
    -   Supported input format:
        -   mol/pdb
        -   SMILES code
    -   Supported output format:
        -   LAMMPS(.lmp)
        -   GROMACS(.gro, .itp)
4.  Write OPLS-AA forcefield file from Maestro
    -   Supported input format:
        -   Any [format that Maestro support]
    -   Supported output format:
        -   LAMMPS(.lmp)
        -   Others pending\...
5.  Packmol wrapper
    -   Supported input format:
        -   xyz
        -   Others pending\...
6.  Basic simulation properties
    -   Initial box dimension
    -   Equilibrium box dimension
    -   Salt concentration
7.  Conductivity analysis
    -   Green--Kubo conductivity
    -   Nernst--Einstein conductivity
8.  Coordination analysis
    -   The distribution of the coordination number of single species
    -   The integral of radial distribution function (The average
        coordination numbers of multiple species)
    -   Solvation structure write out
    -   Population of solvent separated ion pairs (SSIP), contact ion
        pairs (CIP), and aggregates (AGG)
    -   The trajectory (distance) of cation and coordinating species as
        a function of time
    -   The hopping frequency of cation between binding sites
    -   The distribution heat map of cation around binding sites
    -   The averaged nearest neighbor distance of a species
9.  Diffusion analysis
    -   The mean square displacement of all species
    -   The mean square displacement of coordinated species and
        uncoordinated species, separately
    -   Self-diffusion coefficients
10.  Residence time analysis
    -   The residence time of all species

  [format that Maestro support]: https://www.schrodinger.com/kb/1278
