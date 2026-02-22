"""
Jordan Algebra Module - The Albert Algebra J_3(O)

The Albert algebra is the exceptional Jordan algebra of 3x3 Hermitian
matrices over the Octonions. F4 is its automorphism group.

Structure:
    J_3(O) = {X ∈ M_3(O) : X = X†}

    X = | ξ_1    x_3    x̄_2  |
        | x̄_3    ξ_2    x_1  |
        | x_2    x̄_1    ξ_3  |

    where ξ_i ∈ R and x_i ∈ O (octonions)

Key operations:
    - Jordan product: X ∘ Y = (XY + YX) / 2
    - Trace: tr(X) = ξ_1 + ξ_2 + ξ_3
    - Norm: N(X) = det(X)

The "Jordan trace" of an F4 root determines its role in the
prime gap crystalline structure.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


# === Octonion Arithmetic ===

class Octonion:
    """
    Octonion number: a_0 + a_1*e_1 + ... + a_7*e_7

    The octonions are the largest normed division algebra.
    They are non-associative but alternative.
    """

    # Octonion multiplication table (Cayley-Dickson construction)
    # e_i * e_j = MULT_TABLE[i][j] * e_{|result|}
    # Sign is encoded in MULT_SIGNS
    MULT_TABLE = np.array([
        [0, 1, 2, 3, 4, 5, 6, 7],  # e_0 = 1
        [1, 0, 3, 2, 5, 4, 7, 6],  # e_1
        [2, 3, 0, 1, 6, 7, 4, 5],  # e_2
        [3, 2, 1, 0, 7, 6, 5, 4],  # e_3
        [4, 5, 6, 7, 0, 1, 2, 3],  # e_4
        [5, 4, 7, 6, 1, 0, 3, 2],  # e_5
        [6, 7, 4, 5, 2, 3, 0, 1],  # e_6
        [7, 6, 5, 4, 3, 2, 1, 0],  # e_7
    ])

    MULT_SIGNS = np.array([
        [+1, +1, +1, +1, +1, +1, +1, +1],
        [+1, -1, +1, -1, +1, -1, -1, +1],
        [+1, -1, -1, +1, +1, +1, -1, -1],
        [+1, +1, -1, -1, +1, -1, +1, -1],
        [+1, -1, -1, -1, -1, +1, +1, +1],
        [+1, +1, -1, +1, -1, -1, +1, -1],
        [+1, +1, +1, -1, -1, -1, -1, +1],
        [+1, -1, +1, +1, -1, +1, -1, -1],
    ])

    def __init__(self, coeffs=None):
        """Initialize octonion from 8 real coefficients."""
        if coeffs is None:
            self.a = np.zeros(8)
        else:
            self.a = np.array(coeffs, dtype=np.float64)

    @classmethod
    def from_real(cls, x: float) -> 'Octonion':
        """Create real octonion (only e_0 component)."""
        o = cls()
        o.a[0] = x
        return o

    @classmethod
    def basis(cls, i: int) -> 'Octonion':
        """Create basis octonion e_i."""
        o = cls()
        o.a[i] = 1.0
        return o

    def __add__(self, other: 'Octonion') -> 'Octonion':
        return Octonion(self.a + other.a)

    def __sub__(self, other: 'Octonion') -> 'Octonion':
        return Octonion(self.a - other.a)

    def __mul__(self, other: 'Octonion') -> 'Octonion':
        """Octonion multiplication (non-associative!)."""
        result = np.zeros(8)
        for i in range(8):
            for j in range(8):
                k = self.MULT_TABLE[i, j]
                sign = self.MULT_SIGNS[i, j]
                result[k] += sign * self.a[i] * other.a[j]
        return Octonion(result)

    def __rmul__(self, scalar: float) -> 'Octonion':
        return Octonion(scalar * self.a)

    def conjugate(self) -> 'Octonion':
        """Octonion conjugate: ā = a_0 - a_1*e_1 - ... - a_7*e_7"""
        conj = np.copy(self.a)
        conj[1:] = -conj[1:]
        return Octonion(conj)

    def norm_squared(self) -> float:
        """Squared norm: |a|² = a * ā"""
        return np.dot(self.a, self.a)

    def norm(self) -> float:
        """Norm: |a| = √(a * ā)"""
        return np.sqrt(self.norm_squared())

    def real(self) -> float:
        """Real part (e_0 coefficient)."""
        return self.a[0]

    def imag(self) -> np.ndarray:
        """Imaginary part (e_1 through e_7)."""
        return self.a[1:]

    def __repr__(self):
        terms = [f"{self.a[0]:.3f}"]
        for i in range(1, 8):
            if abs(self.a[i]) > 1e-10:
                terms.append(f"{self.a[i]:+.3f}e{i}")
        return "".join(terms)


# === Albert Algebra (J_3(O)) ===

@dataclass
class AlbertElement:
    """
    Element of the Albert algebra J_3(O).

    A 3x3 Hermitian matrix over octonions:
        | ξ_1    x_3    x̄_2  |
        | x̄_3    ξ_2    x_1  |
        | x_2    x̄_1    ξ_3  |

    Stored as: diag = [ξ_1, ξ_2, ξ_3] and off_diag = [x_1, x_2, x_3]
    """
    diag: np.ndarray      # [ξ_1, ξ_2, ξ_3] ∈ R³
    off_diag: Tuple[Octonion, Octonion, Octonion]  # [x_1, x_2, x_3] ∈ O³


def albert_trace(X: AlbertElement) -> float:
    """
    Trace of an Albert algebra element.

    tr(X) = ξ_1 + ξ_2 + ξ_3

    This is the fundamental linear form on J_3(O).
    """
    return np.sum(X.diag)


def albert_inner_product(X: AlbertElement, Y: AlbertElement) -> float:
    """
    Inner product: <X, Y> = tr(X ∘ Y)

    For the standard basis, this gives the Euclidean structure.
    """
    # Simplified: just diagonal contribution
    return np.dot(X.diag, Y.diag)


def albert_algebra_product(X: AlbertElement, Y: AlbertElement) -> AlbertElement:
    """
    Jordan product: X ∘ Y = (XY + YX) / 2

    This is commutative but non-associative.
    """
    # Diagonal part: ξ_i * η_i + Re(x_j * ȳ_j + x_k * ȳ_k)
    new_diag = np.zeros(3)

    for i in range(3):
        new_diag[i] = X.diag[i] * Y.diag[i]

        # Off-diagonal contributions (indices j, k ≠ i)
        j = (i + 1) % 3
        k = (i + 2) % 3

        # x_j * ȳ_j contribution
        prod_j = X.off_diag[j] * Y.off_diag[j].conjugate()
        new_diag[i] += prod_j.real()

        # x_k * ȳ_k contribution
        prod_k = X.off_diag[k] * Y.off_diag[k].conjugate()
        new_diag[i] += prod_k.real()

    # Off-diagonal part (simplified)
    new_off = []
    for i in range(3):
        j = (i + 1) % 3
        k = (i + 2) % 3

        # x_i' = (ξ_j + ξ_k)/2 * (x_i + y_i) + ...
        coeff = (X.diag[j] + X.diag[k] + Y.diag[j] + Y.diag[k]) / 4
        new_x = coeff * (X.off_diag[i].a[0] + Y.off_diag[i].a[0])
        new_off.append(Octonion.from_real(new_x))

    return AlbertElement(new_diag, tuple(new_off))


# === Jordan Trace for F4 Roots ===

class JordanTrace:
    """
    Compute the Jordan trace of F4 roots.

    The Jordan trace maps F4 roots to the Albert algebra,
    revealing their role in the prime gap structure.

    For an F4 root α ∈ R⁴, the Jordan trace is:
        J(α) = Σᵢ αᵢ (when α is in the Cartan subalgebra)

    For the octonionic interpretation:
        J(α) = tr(proj_J(α))

    where proj_J maps to the diagonal of J_3(O).
    """

    def __init__(self):
        # Projection matrix from R^4 to diagonal of J_3(O)
        # Maps F4 Cartan subalgebra to Albert algebra diagonal
        self.proj_matrix = np.array([
            [1, 0, 0, 0],   # ξ_1 component
            [0, 1, 0, 0],   # ξ_2 component
            [0, 0, 1, 1],   # ξ_3 component (combines last two)
        ])

    def __call__(self, root: np.ndarray) -> float:
        """
        Compute Jordan trace of an F4 root.

        Parameters
        ----------
        root : np.ndarray
            F4 root vector in R^4

        Returns
        -------
        float
            Jordan trace value
        """
        # Project to J_3(O) diagonal
        if len(root) == 8:
            root = root[:4]  # Extract F4 part from E8

        diag = self.proj_matrix @ root
        return np.sum(diag)

    def trace_decomposition(self, root: np.ndarray) -> np.ndarray:
        """
        Get the full 3-component diagonal projection.

        Returns [ξ_1, ξ_2, ξ_3] in J_3(O).
        """
        if len(root) == 8:
            root = root[:4]
        return self.proj_matrix @ root

    def classify_by_trace(self, root: np.ndarray) -> str:
        """
        Classify F4 root by its Jordan trace value.

        Categories:
        - "primitive": |J| < 0.5 (near-zero trace)
        - "idempotent": |J| ≈ 1 (unit trace)
        - "nilpotent": J = 0 exactly
        - "regular": other
        """
        j_trace = self(root)

        if abs(j_trace) < 0.01:
            return "nilpotent"
        elif abs(j_trace) < 0.5:
            return "primitive"
        elif abs(abs(j_trace) - 1.0) < 0.1:
            return "idempotent"
        else:
            return "regular"


def jordan_trace_coloring(f4_roots: np.ndarray) -> np.ndarray:
    """
    Compute Jordan trace values for coloring F4-filtered primes.

    Parameters
    ----------
    f4_roots : np.ndarray
        Array of F4 root vectors, shape (n, 4)

    Returns
    -------
    np.ndarray
        Jordan trace values for each root
    """
    jt = JordanTrace()
    return np.array([jt(root) for root in f4_roots])


# === Octonionic Structure Constants ===

def octonion_structure_constants() -> np.ndarray:
    """
    Return the structure constants f_ijk of the octonions.

    e_i * e_j = f_ijk * e_k (summed over k)

    These determine the non-associativity of the Albert algebra.
    """
    f = np.zeros((8, 8, 8))

    # Build from multiplication table
    for i in range(8):
        for j in range(8):
            k = Octonion.MULT_TABLE[i, j]
            sign = Octonion.MULT_SIGNS[i, j]
            f[i, j, k] = sign

    return f


def cayley_plane_projection(root: np.ndarray) -> Tuple[float, float]:
    """
    Project F4 root onto the Cayley plane OP².

    The Cayley plane is the projective plane over octonions,
    a 16-dimensional manifold. F4 acts on it.

    Returns (r, θ) in polar coordinates on a 2D slice.
    """
    if len(root) == 8:
        root = root[:4]

    # Use first two components as complex number
    z = complex(root[0], root[1])
    # Use last two as another coordinate
    w = complex(root[2], root[3])

    # Combine into Cayley plane coordinate
    r = np.sqrt(abs(z)**2 + abs(w)**2)
    theta = np.arctan2(z.imag + w.imag, z.real + w.real)

    return (r, theta)


if __name__ == "__main__":
    # Test octonion arithmetic
    print("=== Octonion Tests ===")
    e1 = Octonion.basis(1)
    e2 = Octonion.basis(2)
    e3 = Octonion.basis(3)

    print(f"e1 * e2 = {e1 * e2}")
    print(f"e2 * e1 = {e2 * e1}")
    print(f"e1 * e2 ≠ e2 * e1: {not np.allclose((e1*e2).a, (e2*e1).a)}")

    # Test Jordan trace
    print("\n=== Jordan Trace Tests ===")
    jt = JordanTrace()

    # Some F4 roots
    roots = [
        np.array([1, 1, 0, 0]),      # Long root
        np.array([1, 0, 0, 0]),      # Short root
        np.array([0.5, 0.5, 0.5, 0.5]),  # Half-integer root
        np.array([1, -1, 0, 0]),     # Another long root
    ]

    for root in roots:
        trace = jt(root)
        decomp = jt.trace_decomposition(root)
        category = jt.classify_by_trace(root)
        print(f"Root {root} -> J={trace:.3f}, decomp={decomp}, type={category}")
