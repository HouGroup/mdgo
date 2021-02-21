Introduction
============

Welcome to the documentation site for mdgo! Mdgo is an python toolkit for classical molecualr dynamics (MD) simulation setup and results analysis, especially for electrolyte systems. The purpose of making this package is for supporting a high-throughput workflow for screening novel electrolytes for battery use. Currently, the package is under active development.

Features
------------

#. Retriving compound structure and information from PubChem

   * Supported searching text:

     * cid, name, smiles, inchi, inchikey or formula

   * Supported output format:

     * XML, ASNT/B, JSON, SDF, CSV, PNG, TXT

#. Write OPLS-AA forcefield file from LigParGen

   * Supported input format:

     * mol/pdb

     * SMILES code

   * Supported output format:

     * LAMMPS(.lmp)

     * GROMACS(.gro, .itp)

#. Write OPLS-AA forcefield file from Maestro

   * Supported input format:

     * Any `format that Maestro support <https://www.schrodinger.com/kb/1278>`_

   * Supported output format:

     * LAMMPS(.lmp)

     * Others pending...

#. Packmol wrapper

   * Supported input format:

     * xyz

     * Others pending...

#. Basic simulation properties

   * Initial box dimension

   * Equilibrium box dimension

   * Salt concentration

#. Conductivity analysis

   * Green–Kubo conductivity

   * Nernst–Einstein conductivity

#. Coordination analysis

   * The distribution of the coordination number of single species

   * The integral of radial distribution function (The average coordination numbers of multiple species)

   * Solvation structure write out

   * Population of solvent separated ion pairs (SSIP), contact ion pairs (CIP), and aggregates (AGG)

   * The trajectory (distance) of cation and coordinating species as a function of time

   * The hopping frequency of cation between binding sites

   * The distribution heat map of cation around binding sites

   * The averaged nearest neighbor distance of a species

#. Diffusion analysis

   * The mean square displacement of all species

   * The mean square displacement of coordinated species and uncoordinated species, separately

   * Self-diffusion coefficients

#. Residence time analysis

   * The residence time of all species



Installation
------------

Requirements
^^^^^^^^^^^^
mdgo requires numpy, pandas, matplotlib, scipy, tqdm, statsmodels, pymatgen, pubchempy, selenium, MDAnalysis (version 2.0.0-dev0 prefered) and their dependencies.

Getting source code
^^^^^^^^^^^^^^^^^^^

If not available already, use the following steps.

1. Install `git <https://git-scm.com>`_ if not already packaged with your system.

2. Download the mdgo source code using the command::

    git clone https://github.com/htz1992213/mdgo.git

Installation from source
^^^^^^^^^^^^^^^^^^^^^^^^
1. Navigate to mdgo root directory::

    cd mdgo

2. Install the code, using the command::

    pip install .

3. The latest version MDAnalysis==2.0.0.dev0 is recommended, you may download the source code of the latest MDAnalysis from github and install using pip to replace an existing version.

Installation in development mode
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
1. Navigate to mdgo root directory::

    cd mdgo

2. Install the code in "editable" mode, using the command::

    pip install -e .

3. The latest version MDAnalysis==2.0.0.dev0 is recommended, you may download the source code of the latest MDAnalysis from github and install using pip to replace an existing version.

Contributing
------------

Reporting bugs
^^^^^^^^^^^^^^

Please report any bugs and issues at mdgo's
`Github Issues page <https://github.com/htz1992213/mdgo/issues>`_.

Developing new functionality
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You may submit new code/bugfixes by sending a pull request to the mdgo's github repository. 

How to cite mdgo
----------------

pending...

License
-------

Mdgo is released under the MIT License. The terms of the license are as
follows:

.. literalinclude:: ../../LICENSE.rst

About the Team
--------------

Tingzheng Hou started mdgo in 2020 under the supervision of Prof. Kristin Persson at University of California, berkeley. 

Copyright Policy
----------------

The following banner should be used in any source code file
to indicate the copyright and license terms::

    # Copyright (c) Tingzheng Hou.
    # Distributed under the terms of the MIT License.

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
