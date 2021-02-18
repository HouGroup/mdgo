Introduction
============

Welcome to the documentation site for mdgo! Mdgo is an python toolkit for classical molecualr dynamics (MD) simulation setup and results analysis, especially for electrolyte systems. The purpose of making this package is for supporting a high-throughput workflow for screening novel electrolytes for battery use. Currently, the package is under active development.

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

Installation in development mode
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
1. Navigate to mdgo root directory::

    cd mdgo

2. Install the code, using the command::

    pip install -e .




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
