"""
G2 Root Lattice - Smallest Exceptional Lie Group

The G2 root system has 12 roots in R^2, with a 3:1 long-to-short
length ratio (unique among simple Lie groups).

G2 Facts:
- Dimension: 14 (as Lie algebra)
- Rank: 2
- Number of roots: 12 (6 short + 6 long)
- Cartan matrix: [[2,-1],[-3,2]]
- Automorphism group of the octonions

The 12 G2 roots in R^2:
- Short (6, norm 1): at angles 0, 60, 120, 180, 240, 300 degrees
- Long (6, norm sqrt(3)): at angles 30, 90, 150, 210, 270, 330 degrees

E8→G2 mapping: project E8 roots to first 2 coordinates,
then cosine similarity in R^2.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class G2RootData:
    """Container for G2 root analysis results"""
    g2_roots: np.ndarray          # The 12 G2 roots in R^2
    e8_to_g2_map: Dict[int, int]  # Maps E8 root index to G2 root index
    cartan_matrix: np.ndarray     # 2x2 G2 Cartan matrix
    root_lengths: np.ndarray      # 1.0 (short) or sqrt(3) (long)


class G2Lattice:
    """
    The G2 root lattice, linked to E8 via projection.

    G2 roots live in R^2. The E8→G2 mapping projects E8 roots
    onto their first 2 coordinates and finds the nearest G2 root
    by cosine similarity.
    """

    def __init__(self, e8_lattice=None):
        """
        Initialize G2 lattice, optionally linking to parent E8.

        Parameters
        ----------
        e8_lattice : E8Lattice, optional
            Parent E8 lattice for cross-referencing
        """
        self.e8 = e8_lattice

        # Generate G2 roots
        self.roots_2d = self._generate_g2_roots_2d()

        # Build mapping to E8 if available
        if e8_lattice is not None:
            self.e8_to_g2, self.g2_to_e8 = self._build_e8_mapping()
        else:
            self.e8_to_g2 = {}
            self.g2_to_e8 = {}

        # Cartan matrix
        self.cartan_matrix = self._g2_cartan_matrix()
        self.eigenvalues = np.linalg.eigvalsh(self.cartan_matrix)

        # Root classification
        self.long_root_indices = [i for i in range(12) if self._is_long(i)]
        self.short_root_indices = [i for i in range(12) if not self._is_long(i)]

        # Precompute for efficiency
        self._precompute_characters()

    def _generate_g2_roots_2d(self) -> np.ndarray:
        """
        Generate all 12 G2 roots in R^2.

        Short roots (6, norm 1): at angles k*60 degrees, k=0,...,5
        Long roots (6, norm sqrt(3)): at angles (k*60+30) degrees, k=0,...,5

        Returns
        -------
        np.ndarray
            Shape (12, 2) array of G2 root vectors
        """
        roots = []

        # Short roots (norm 1)
        for k in range(6):
            angle = k * np.pi / 3.0
            roots.append([np.cos(angle), np.sin(angle)])

        # Long roots (norm sqrt(3))
        for k in range(6):
            angle = (k * 60.0 + 30.0) * np.pi / 180.0
            roots.append([np.sqrt(3.0) * np.cos(angle),
                          np.sqrt(3.0) * np.sin(angle)])

        roots = np.array(roots)
        assert len(roots) == 12, f"BUG: generated {len(roots)} G2 roots, expected 12"
        return roots

    def _is_long(self, idx: int) -> bool:
        """Check if a G2 root is long (norm sqrt(3)) vs short (norm 1)."""
        return np.linalg.norm(self.roots_2d[idx]) > 1.2

    def _build_e8_mapping(self) -> Tuple[Dict[int, int], Dict[int, int]]:
        """
        Build E8→G2 mapping via projection to first 2 coords + cosine similarity.

        Since G2 lives in R^2 and E8 in R^8, this is a genuine projection.
        Quality threshold is 0.7 (same as F4).
        """
        e8_to_g2 = {}
        g2_to_e8 = {}

        # Precompute normalized G2 roots
        g2_norms = np.linalg.norm(self.roots_2d, axis=1, keepdims=True)
        g2_normalized = self.roots_2d / (g2_norms + 1e-10)

        for e8_idx, e8_root in enumerate(self.e8.roots):
            # Project E8 root to first 2 coordinates
            proj = e8_root[:2]
            proj_norm = np.linalg.norm(proj)

            if proj_norm < 0.01:
                continue

            proj_normalized = proj / proj_norm
            similarities = np.dot(g2_normalized, proj_normalized)
            g2_idx = int(np.argmax(np.abs(similarities)))
            best_sim = abs(similarities[g2_idx])

            e8_to_g2[e8_idx] = g2_idx
            if g2_idx not in g2_to_e8:
                g2_to_e8[g2_idx] = e8_idx

        return e8_to_g2, g2_to_e8

    def _g2_cartan_matrix(self) -> np.ndarray:
        """
        The G2 Cartan matrix.

        The off-diagonal -3 reflects the 3:1 length ratio squared.
        """
        return np.array([
            [ 2, -1],
            [-3,  2],
        ], dtype=np.float64)

    def _precompute_characters(self):
        """
        Precompute G2 character values for each root.

        G2 has a 14-dimensional adjoint representation.
        Long roots (norm sqrt(3)) get character 2.0,
        short roots (norm 1) get character 1.0.
        """
        self.characters = np.zeros(12)

        for i, root in enumerate(self.roots_2d):
            weyl_height = np.sum(np.abs(root))
            if self._is_long(i):
                self.characters[i] = 2.0 * (1 + 0.1 * weyl_height)
            else:
                self.characters[i] = 1.0 * (1 + 0.1 * weyl_height)

    # === Public Methods ===

    def is_g2_root(self, e8_index: int, threshold: float = 0.7) -> bool:
        """
        Check if an E8 root has strong G2 character.

        Uses cosine similarity between E8→R^2 projection and nearest G2 root.
        """
        if e8_index not in self.e8_to_g2:
            return False

        if self.e8 is None:
            return True

        e8_root = self.e8.roots[e8_index]
        proj = e8_root[:2]
        proj_norm = np.linalg.norm(proj)

        if proj_norm < 0.01:
            return False

        g2_idx = self.e8_to_g2[e8_index]
        g2_root = self.roots_2d[g2_idx]
        g2_norm = np.linalg.norm(g2_root)

        if g2_norm < 0.01:
            return False

        similarity = abs(np.dot(proj, g2_root)) / (proj_norm * g2_norm)
        return similarity >= threshold

    def project_e8_to_g2(self, e8_index: int) -> Optional[int]:
        """Project an E8 root to its G2 component. Returns None if not mapped."""
        return self.e8_to_g2.get(e8_index, None)

    def get_projection_quality(self, e8_index: int) -> float:
        """Get the quality of E8→G2 projection (0 to 1)."""
        if e8_index not in self.e8_to_g2 or self.e8 is None:
            return 0.0

        e8_root = self.e8.roots[e8_index]
        proj = e8_root[:2]
        proj_norm = np.linalg.norm(proj)

        if proj_norm < 0.01:
            return 0.0

        g2_idx = self.e8_to_g2[e8_index]
        g2_root = self.roots_2d[g2_idx]
        g2_norm = np.linalg.norm(g2_root)

        if g2_norm < 0.01:
            return 0.0

        return abs(np.dot(proj, g2_root)) / (proj_norm * g2_norm)

    def get_g2_root(self, g2_index: int) -> np.ndarray:
        """Get the G2 root vector (in R^2) by index."""
        return self.roots_2d[g2_index]

    def get_character(self, g2_index: int) -> float:
        """Get the G2 character value for a root."""
        return self.characters[g2_index]

    def root_norm(self, g2_index: int) -> float:
        """Get the norm of a G2 root."""
        return np.linalg.norm(self.roots_2d[g2_index])

    def is_long_root(self, g2_index: int) -> bool:
        """Check if root is long (norm sqrt(3)) vs short (norm 1)."""
        return g2_index in self.long_root_indices

    def jordan_trace(self, g2_index: int) -> float:
        """
        Compute the trace-2 (sum of 2 coords) for a G2 root.
        This is the natural trace for G2 in R^2.
        """
        return np.sum(self.roots_2d[g2_index])

    def compute_g2_spectrum(self, e8_root_assignments: np.ndarray) -> np.ndarray:
        """
        Compute the G2 power spectrum from E8 root assignments.

        Parameters
        ----------
        e8_root_assignments : np.ndarray
            Array of E8 root indices (from prime gap analysis)

        Returns
        -------
        np.ndarray
            Shape (12,) array of G2 root frequencies
        """
        g2_counts = np.zeros(12)

        for e8_idx in e8_root_assignments:
            g2_idx = self.project_e8_to_g2(int(e8_idx))
            if g2_idx is not None:
                g2_counts[g2_idx] += 1

        return g2_counts

    def summary(self) -> str:
        """Return a summary of the G2 lattice."""
        lines = [
            "G2 Root Lattice Summary",
            "=" * 40,
            f"Total roots: {len(self.roots_2d)}",
            f"Long roots (norm sqrt(3)): {len(self.long_root_indices)}",
            f"Short roots (norm 1): {len(self.short_root_indices)}",
            f"Cartan matrix eigenvalues: {self.eigenvalues}",
        ]

        if self.e8:
            n_mapped = len(self.e8_to_g2)
            n_good = sum(1 for e8_idx in self.e8_to_g2 if self.is_g2_root(e8_idx))
            lines.append(f"E8 roots mapping to G2: {n_mapped}/240")
            lines.append(f"  Quality >= 0.7: {n_good}")

        return "\n".join(lines)


if __name__ == "__main__":
    # Test G2 lattice standalone
    g2 = G2Lattice()
    print(g2.summary())

    print("\nAll 12 G2 roots:")
    for i in range(12):
        root = g2.get_g2_root(i)
        norm = g2.root_norm(i)
        trace = g2.jordan_trace(i)
        char = g2.get_character(i)
        rtype = "long" if g2.is_long_root(i) else "short"
        print(f"  {i}: [{root[0]:+.4f}, {root[1]:+.4f}] "
              f"(norm={norm:.3f}, {rtype}, trace2={trace:.3f}, chi={char:.3f})")
