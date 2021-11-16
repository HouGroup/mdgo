
================================
Force Fields for Aqueous Systems
================================

The Aqueous module provides tools for setting up molecular dynamics simulations
involving water and ions.

Water Models
============

`mdgo` contains parameters for several popular water models. This section lists
a brief description and literature reference to the available models.

SPC
---

TIP3P-EW
--------

TIP3P-FB
--------

Wang, L., Martinez, T. J., Pande, V.S., Building Force Fields: An Automatic, Systematic,
and Reproducible Approach. J. Phys. Chem. Lett. 2014, 5, 11, 1885–1891.
https://pubs.acs.org/doi/abs/10.1021/jz500737m

Parameters are given in Supporting Table 1. Note that the epsilon for Oxygen must be converted
from kJ/mol to kcal/mol.

TIP4P-EW
--------

[Vega & de Miguel, J Chem Phys 126:154707 (2007), Vega et al, Faraday Discuss 141:251 (2009)].

TIP4P-FB
--------

Wang, L., Martinez, T. J., Pande, V.S., Building Force Fields: An Automatic, Systematic,
and Reproducible Approach. J. Phys. Chem. Lett. 2014, 5, 11, 1885–1891.
https://pubs.acs.org/doi/abs/10.1021/jz500737m

Parameters are given in Supporting Table 1. Note that the epsilon for oxygen must be converted
from kJ/mol to kcal/mol.

TIP4P-2005
----------

Abascal & Vega, J Chem Phys 123:234505 (2005)

OPC
----

Izadi, Anandakrishnan, and Onufriev, Building Water Models: A Different Approach. 
J. Phys. Chem. Lett. 2014, 5, 21, 3863–3871 https://doi.org/10.1021/jz501780a

Parameters are given in Table 2. Note that the epsilon for oxygen must be converted
from kJ/mol to kcal/mol.

OPC3
----

Izadi and Onufriev, Accuracy limit of rigid 3-point water models
J. Chemical Physics 145, 074501, 2016. https://doi.org/10.1063/1.4960175

Parameters are given in Table II. Note that the epsilon for oxygen must be converted
from kJ/mol to kcal/mol.


Ion Parameter Sets
==================

``mdgo`` contains a compilation of several sets of Lennard Jones
parameters for ions in water. All values are reported as :math:`\sigma_i`
and :math:`\epsilon_i` in the equation

.. math::

   E = 4 \\epsilon_i \\left[ \\left( \\frac{\sigma_i}{r} \\right)^{12} - \\left( \\frac{\sigma_i}{r} \\right)^{6} \\right]

Values of :math:`\sigma_i` and :math:`\epsilon_i` are given in Angstrom
and kcal/mol, respectively, corresponding to the ‘real’ units system in
LAMMPS.

Aqvist (aq)
-----------

Aqvist, J. Ion-Water Interaction Potentials Derived from Free Energy
Perturbation Slmulations J. Phys. Chem. 1990, 94, 8021– 8024.
https://pubs.acs.org/doi/10.1021/j100384a009

Values were parameterized to the SPC water model and are reported in
Table I and II as :math:`A_i` and :math:`B_i` coefficients in the
following form of the Lennard-Jones potential:

.. math::


   E = \left[ \left( \frac{A_i^2}{r} \right)^{12} - \left( \frac{B_i^2}{r} \right)^{6} \right]

This parameter set is a work in progress!

Jensen and Jorgensen (jj)
-------------------------

Jensen, K. P. and Jorgensen, W. L., Halide, Ammonium, and Alkali Metal
Ion Parameters for Modeling Aqueous Solutions. J. Chem. Theory Comput.
2006, 2, 6, 1499–1509. https://pubs.acs.org/doi/abs/10.1021/ct600252r

Values were parameterized to the TIP4P water model using geometric
combining rules and are reported directly as sigma_i and epsilon_i in
Table 2.

Joung-Cheatham (jc)
-------------------

Joung, and Thomas E. Cheatham, Thomas E. III, Determination of Alkali
and Halide Monovalent Ion Parameters for Use in Explicitly Solvated
Biomolecular Simulations. J. Phys. Chem. B 2008, 112, 30, 9020–9041.
https://pubs.acs.org/doi/10.1021/jp8001614

Values were parameterized for the SPC/E, TIP3P, and TIP4P_EW water
models using Lorentz-Berthelot combining rules (LAMMPS: ‘arithmetic’)
and are reported in Table 5 as :math:`R_{min}`/2 and epsilon_i. R_min/2
values are converted to :math:`\sigma_i` values using
:math:`\sigma_i = R_{min}/2 * 2^(5/6)`

Li and Merz group (lm)
----------------------

Sengupta et al. Parameterization of Monovalent Ions for the OPC3, OPC,
TIP3P-FB, and TIP4P-FB Water Models. J. Chem. Information Modeling
61(2), 2021. https://pubs.acs.org/doi/10.1021/acs.jcim.0c01390

Li et al. Systematic Parametrization of Divalent Metal Ions for the
OPC3, OPC, TIP3P-FB, and TIP4P-FB Water Models. J. Chem. Theory and
Computation 16(7), 2020.
https://pubs.acs.org/doi/10.1021/acs.jctc.0c00194

Li et al. Parametrization of Trivalent and Tetravalent Metal Ions for
the OPC3, OPC, TIP3P-FB, and TIP4P-FB Water Models. J. Chem. Theory and
Computation 17(4), 2021.
https://pubs.acs.org/doi/10.1021/acs.jctc.0c01320

Values were parameterized for the OPC, OPC3, TIP3P-FB, and TIP4P-FB
water models using Lorentz-Berthelot combining rules (LAMMPS:
‘arithmetic’) and are reported in Table 3 as :math:`R_{min}`/2 and
epsilon_i. R_min/2 values are converted to :math:`\sigma_i` values using
:math:`\sigma_i = R_{min}/2 * 2^(5/6)`. This set of values is optimized
for reproducing ion-oxygen distance. An alternate set of values optimized for
hydration free energies is available in the original papers.

