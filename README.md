# mdgo

A code base for classical molecualr dynamics (MD) simulation setup and results analysis. 

Requirements
------------

mdgo requires numpy, pandas, matplotlib, scipy, tqdm, statsmodels, pymatgen, pubchempy, selenium, MDAnalysis (version 2.0.0-dev0 prefered) and their dependencies.                          

Source Code
------------
If not available already, use the following steps.

1. Install [git](http://git-scm.com), if not already packaged with your system.

2. Download the mdgo source code using the command::

    `git clone https://github.com/htz1992213/mdgo.git`
    
Installation
------------
1. Navigate to mdgo root directory:

    `cd mdgo`

2. Install the code, using the command:

    `pip install .`

Installation in development mode
------------
1. Navigate to mdgo root directory:

    `cd mdgo`

2. Install the code, using the command:

    `pip install -e .`


Features
------------

1. Write OPLS-AA forcefield file from LigParGen
   - Supported input format: 
      - mol/pdb
      - SMILES code
   - Supported output format:
      - LAMMPS(.lmp)
      - GROMACS(.gro, .itp)
2. Write OPLS-AA forcefield file from Maestro
   - Supported input format: 
      - Any format that Maestro support 
      - See https://www.schrodinger.com/kb/1278
   - Supported output format:
      - LAMMPS(.lmp)
      - Others pending... 
3. Packmol wrapper
   - Supported input format: 
      - xyz
      - Others pending...   
4. Basic simulation properties
   - Initial box dimension
   - Equilibrium box dimension
   - Salt concentration
5. Conductivity analysis
   - Green–Kubo conductivity
   - Nernst–Einstein conductivity
6. Coordination analysis
   - The distribution of the coordination number of single species
   - The integral of radial distribution function (The average coordination numbers of multiple species)
   - Solvation structure write out
   - Population of solvent separated ion pairs (SSIP), contact ion pairs (CIP), and aggregates (AGG)
   - The trajectory (distance) of cation and coordinating species as a function of time
   - The hopping frequency of cation between binding sites
   - The distribution heat map of cation around binding sites
   - The averaged nearest neighbor distance of a species
7. Diffusion analysis
   - The mean square displacement of all species
   - The mean square displacement of coordinated species and uncoordinated species, separately
   - Self-diffusion coefficients
8. Residence time analysis
   - The residence time of all species
