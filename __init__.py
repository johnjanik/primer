"""
F4 Tuning Module for Prime Analysis

Isolates the 52-dimensional F4 sub-harmonic from the E8 signal,
revealing the Jordan-algebraic skeleton of the prime distribution.

Based on the Tits-Kantor-Koecher construction:
    G2 ⊂ F4 ⊂ E8

Where F4 = Aut(J_3(O)) is the automorphism group of the Albert algebra.
"""

from .f4_lattice import F4Lattice
from .salem_jordan import SalemJordanKernel
from .jordan_algebra import JordanTrace, albert_algebra_product
from .f4_eft import F4ExceptionalFourierTransform

__all__ = [
    'F4Lattice',
    'SalemJordanKernel',
    'JordanTrace',
    'albert_algebra_product',
    'F4ExceptionalFourierTransform'
]
