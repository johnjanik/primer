"""
E6 Root Lattice - Sublattice of E8

The E6 root system is embedded in E8 via the constraint v[0] == v[1] == v[2]
(perpendicular to both e_0-e_1 and e_1-e_2).

E6 Facts:
- Dimension: 78 (as Lie algebra)
- Rank: 6
- Number of roots: 72 (all norm sqrt(2))
- Dynkin diagram: o-o-o(-o)-o-o (branch at node 3)

The 72 E6 roots in R^8 (subset of E8's 240):
- Type I (40): ±e_i±e_j with i,j >= 3 → C(5,2)×4 = 40
- Type II (32): (±½)^8 with even neg AND v[0]==v[1]==v[2] → 16+16
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class E6RootData:
    """Container for E6 root analysis results"""
    e6_roots: np.ndarray          # The 72 E6 roots in R^8
    e8_to_e6_map: Dict[int, int]  # Maps E8 root index to E6 root index
    e6_to_e8_map: Dict[int, int]  # Maps E6 root index to E8 root index
    cartan_matrix: np.ndarray     # 6x6 E6 Cartan matrix
    root_lengths: np.ndarray      # All sqrt(2) for E6


class E6Lattice:
    """
    The E6 root lattice as a sublattice of E8.

    E6 roots are exactly those E8 roots satisfying v[0] == v[1] == v[2].
    Since E6 roots live in R^8 (no projection needed), the E8→E6
    mapping uses direct membership for exact matches and cosine
    similarity in R^8 for non-E6 roots.
    """

    def __init__(self, e8_lattice=None):
        """
        Initialize E6 lattice, optionally linking to parent E8.

        Parameters
        ----------
        e8_lattice : E8Lattice, optional
            Parent E8 lattice for cross-referencing
        """
        self.e8 = e8_lattice

        # Generate E6 roots (filtering E8 by v[0]==v[1]==v[2])
        self.roots_8d = self._generate_e6_roots_8d()

        # Build mapping to E8 if available
        if e8_lattice is not None:
            self.e8_to_e6, self.e6_to_e8 = self._build_e8_mapping()
        else:
            self.e8_to_e6 = {}
            self.e6_to_e8 = {}

        # Cartan matrix and derived quantities
        self.cartan_matrix = self._e6_cartan_matrix()
        self.eigenvalues = np.linalg.eigvalsh(self.cartan_matrix)

        # Root classification (all E6 roots have norm sqrt(2))
        self.long_root_indices = list(range(len(self.roots_8d)))
        self.short_root_indices = []

        # Precompute for efficiency
        self._precompute_characters()

    def _generate_e6_roots_8d(self) -> np.ndarray:
        """
        Generate all 72 E6 roots in R^8 by filtering E8 roots.

        E6 constraint: v[0] == v[1] == v[2]

        Returns
        -------
        np.ndarray
            Shape (72, 8) array of E6 root vectors
        """
        roots = []

        # Type I: ±e_i ± e_j for i < j
        for i in range(8):
            for j in range(i + 1, 8):
                for s1 in [-1, 1]:
                    for s2 in [-1, 1]:
                        root = np.zeros(8)
                        root[i] = s1
                        root[j] = s2
                        # Check v[0] == v[1] == v[2]
                        if (abs(root[0] - root[1]) < 1e-10 and
                            abs(root[1] - root[2]) < 1e-10):
                            roots.append(root)

        # Type II: (±½)^8 with even number of minus signs AND v[0]==v[1]==v[2]
        for mask in range(256):
            signs = [1 if (mask >> i) & 1 else -1 for i in range(8)]
            if sum(1 for s in signs if s == -1) % 2 == 0:
                root = np.array([s * 0.5 for s in signs])
                if (abs(root[0] - root[1]) < 1e-10 and
                    abs(root[1] - root[2]) < 1e-10):
                    roots.append(root)

        roots = np.array(roots)
        assert len(roots) == 72, f"BUG: generated {len(roots)} E6 roots, expected 72"
        return roots

    def _build_e8_mapping(self) -> Tuple[Dict[int, int], Dict[int, int]]:
        """
        Build bidirectional mapping between E8 and E6 root indices.

        For E8 roots that ARE E6 roots (v[0]==v[1]==v[2]): exact membership.
        For other E8 roots: nearest E6 root by cosine similarity in R^8.
        """
        e8_to_e6 = {}
        e6_to_e8 = {}

        # Precompute normalized E6 roots
        e6_norms = np.linalg.norm(self.roots_8d, axis=1, keepdims=True)
        e6_normalized = self.roots_8d / (e6_norms + 1e-10)

        for e8_idx, e8_root in enumerate(self.e8.roots):
            # Check if this E8 root is an E6 root
            if (abs(e8_root[0] - e8_root[1]) < 1e-10 and
                abs(e8_root[1] - e8_root[2]) < 1e-10):
                # Find exact match
                diffs = np.max(np.abs(self.roots_8d - e8_root), axis=1)
                match_idx = np.argmin(diffs)
                if diffs[match_idx] < 1e-10:
                    e8_to_e6[e8_idx] = int(match_idx)
                    if int(match_idx) not in e6_to_e8:
                        e6_to_e8[int(match_idx)] = e8_idx
                    continue

            # Not an E6 root — find nearest by cosine similarity in R^8
            e8_norm = np.linalg.norm(e8_root)
            if e8_norm < 0.01:
                continue
            e8_normalized = e8_root / e8_norm
            similarities = np.dot(e6_normalized, e8_normalized)
            e6_idx = int(np.argmax(np.abs(similarities)))
            e8_to_e6[e8_idx] = e6_idx
            if e6_idx not in e6_to_e8:
                e6_to_e8[e6_idx] = e8_idx

        return e8_to_e6, e6_to_e8

    def _e6_cartan_matrix(self) -> np.ndarray:
        """
        The E6 Cartan matrix.

        Dynkin diagram: o-o-o(-o)-o-o
                        1 2 3  4  5 6
        Branch at node 3.
        """
        return np.array([
            [ 2, -1,  0,  0,  0,  0],
            [-1,  2, -1,  0,  0,  0],
            [ 0, -1,  2, -1,  0, -1],
            [ 0,  0, -1,  2, -1,  0],
            [ 0,  0,  0, -1,  2,  0],
            [ 0,  0, -1,  0,  0,  2],
        ], dtype=np.float64)

    def _precompute_characters(self):
        """
        Precompute E6 character values for each root.

        E6 has a 78-dimensional adjoint representation.
        All E6 roots are long (norm sqrt(2)).
        """
        self.characters = np.zeros(len(self.roots_8d))

        for i, root in enumerate(self.roots_8d):
            weyl_height = np.sum(np.abs(root))
            # All E6 roots are long → base character 2.0
            self.characters[i] = 2.0 * (1 + 0.1 * weyl_height)

    # === Public Methods ===

    def is_e6_root(self, e8_index: int, threshold: float = 0.7) -> bool:
        """
        Check if an E8 root has strong E6 character.

        For exact E6 roots (v[0]==v[1]==v[2]): always True.
        For others: cosine similarity threshold.
        """
        if e8_index not in self.e8_to_e6:
            return False

        if self.e8 is None:
            return True

        e8_root = self.e8.roots[e8_index]

        # Exact E6 root?
        if (abs(e8_root[0] - e8_root[1]) < 1e-10 and
            abs(e8_root[1] - e8_root[2]) < 1e-10):
            return True

        # Compute projection quality via cosine similarity in R^8
        e6_idx = self.e8_to_e6[e8_index]
        e6_root = self.roots_8d[e6_idx]

        e8_norm = np.linalg.norm(e8_root)
        e6_norm = np.linalg.norm(e6_root)
        if e8_norm < 0.01 or e6_norm < 0.01:
            return False

        similarity = abs(np.dot(e8_root, e6_root)) / (e8_norm * e6_norm)
        return similarity >= threshold

    def project_e8_to_e6(self, e8_index: int) -> Optional[int]:
        """Project an E8 root to its E6 component. Returns None if not mapped."""
        return self.e8_to_e6.get(e8_index, None)

    def get_projection_quality(self, e8_index: int) -> float:
        """Get the quality of E8→E6 projection (0 to 1)."""
        if e8_index not in self.e8_to_e6 or self.e8 is None:
            return 0.0

        e8_root = self.e8.roots[e8_index]

        # Exact E6 root?
        if (abs(e8_root[0] - e8_root[1]) < 1e-10 and
            abs(e8_root[1] - e8_root[2]) < 1e-10):
            return 1.0

        e6_idx = self.e8_to_e6[e8_index]
        e6_root = self.roots_8d[e6_idx]

        e8_norm = np.linalg.norm(e8_root)
        e6_norm = np.linalg.norm(e6_root)
        if e8_norm < 0.01 or e6_norm < 0.01:
            return 0.0

        return abs(np.dot(e8_root, e6_root)) / (e8_norm * e6_norm)

    def get_e6_root(self, e6_index: int) -> np.ndarray:
        """Get the E6 root vector (in R^8) by index."""
        return self.roots_8d[e6_index]

    def get_character(self, e6_index: int) -> float:
        """Get the E6 character value for a root."""
        return self.characters[e6_index]

    def root_norm(self, e6_index: int) -> float:
        """Get the norm of an E6 root."""
        return np.linalg.norm(self.roots_8d[e6_index])

    def jordan_trace(self, e6_index: int) -> float:
        """
        Compute the trace-8 (sum of all 8 coords) for an E6 root.
        This is the natural trace for E6 in R^8.
        """
        return np.sum(self.roots_8d[e6_index])

    def compute_e6_spectrum(self, e8_root_assignments: np.ndarray) -> np.ndarray:
        """
        Compute the E6 power spectrum from E8 root assignments.

        Parameters
        ----------
        e8_root_assignments : np.ndarray
            Array of E8 root indices (from prime gap analysis)

        Returns
        -------
        np.ndarray
            Shape (72,) array of E6 root frequencies
        """
        e6_counts = np.zeros(len(self.roots_8d))

        for e8_idx in e8_root_assignments:
            e6_idx = self.project_e8_to_e6(int(e8_idx))
            if e6_idx is not None:
                e6_counts[e6_idx] += 1

        return e6_counts

    def summary(self) -> str:
        """Return a summary of the E6 lattice."""
        lines = [
            "E6 Root Lattice Summary",
            "=" * 40,
            f"Total roots: {len(self.roots_8d)}",
            f"Long roots (norm sqrt(2)): {len(self.long_root_indices)}",
            f"Short roots: {len(self.short_root_indices)}",
            f"Cartan matrix eigenvalues: {self.eigenvalues}",
        ]

        if self.e8:
            n_mapped = len(self.e8_to_e6)
            n_exact = sum(1 for e8_idx in self.e8_to_e6
                         if (abs(self.e8.roots[e8_idx][0] - self.e8.roots[e8_idx][1]) < 1e-10 and
                             abs(self.e8.roots[e8_idx][1] - self.e8.roots[e8_idx][2]) < 1e-10))
            lines.append(f"E8 roots mapping to E6: {n_mapped}/240")
            lines.append(f"  Exact E6 roots: {n_exact}")
            lines.append(f"  Approximate mappings: {n_mapped - n_exact}")

        return "\n".join(lines)


if __name__ == "__main__":
    # Test E6 lattice standalone
    e6 = E6Lattice()
    print(e6.summary())

    print("\nFirst 10 E6 roots:")
    for i in range(10):
        root = e6.get_e6_root(i)
        norm = e6.root_norm(i)
        trace = e6.jordan_trace(i)
        char = e6.get_character(i)
        print(f"  {i}: {root} (norm={norm:.3f}, trace8={trace:.3f}, chi={char:.3f})")
