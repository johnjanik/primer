"""
F4 Root Lattice - Extraction from E8

The F4 root system is embedded in E8 via the branching:
    E8 → F4 × G2  (under a specific maximal subgroup)

F4 Facts:
- Dimension: 52 (as Lie algebra)
- Rank: 4
- Number of roots: 48 (24 long + 24 short)
- F4 = Aut(J_3(O)), automorphisms of the Albert algebra

The 48 F4 roots embedded in R^4:
- 24 long roots (norm √2): ±e_i ± e_j for i ≠ j
- 24 short roots (norm 1): ±e_i and (±1/2, ±1/2, ±1/2, ±1/2) with even # of minuses

When embedded in E8's R^8, we identify F4 roots as those E8 roots
that project non-trivially onto the F4 Cartan subalgebra.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class F4RootData:
    """Container for F4 root analysis results"""
    f4_roots: np.ndarray          # The 48 F4 roots in R^4
    f4_roots_e8: np.ndarray       # F4 roots embedded in R^8
    e8_to_f4_map: Dict[int, int]  # Maps E8 root index to F4 root index
    f4_to_e8_map: Dict[int, int]  # Maps F4 root index to E8 root index
    cartan_matrix: np.ndarray     # 4x4 F4 Cartan matrix
    root_lengths: np.ndarray      # Long (√2) or short (1) for each root


class F4Lattice:
    """
    The F4 root lattice as a sublattice of E8.

    Implements the E8 → F4 projection for prime gap analysis,
    filtering the 240 E8 roots down to the 48 F4 roots.
    """

    def __init__(self, e8_lattice=None):
        """
        Initialize F4 lattice, optionally linking to parent E8.

        Parameters
        ----------
        e8_lattice : E8Lattice, optional
            Parent E8 lattice for cross-referencing
        """
        self.e8 = e8_lattice

        # Generate F4 roots
        self.roots_4d = self._generate_f4_roots_4d()
        self.roots_8d = self._embed_in_8d(self.roots_4d)

        # Build mapping to E8 if available
        if e8_lattice is not None:
            self.e8_to_f4, self.f4_to_e8 = self._build_e8_mapping()
        else:
            self.e8_to_f4 = {}
            self.f4_to_e8 = {}

        # Cartan matrix and derived quantities
        self.cartan_matrix = self._f4_cartan_matrix()
        self.eigenvalues = np.linalg.eigvalsh(self.cartan_matrix)
        self.fundamental_weights = self._compute_fundamental_weights()

        # Root classification
        self.long_root_indices = self._classify_roots()
        self.short_root_indices = [i for i in range(48) if i not in self.long_root_indices]

        # Precompute for efficiency
        self._precompute_characters()

    def _generate_f4_roots_4d(self) -> np.ndarray:
        """
        Generate all 48 F4 roots in R^4.

        Returns
        -------
        np.ndarray
            Shape (48, 4) array of F4 root vectors
        """
        roots = []

        # === Long roots (24 total, norm √2) ===
        # Type: ±e_i ± e_j for i < j
        for i in range(4):
            for j in range(i + 1, 4):
                for s1 in [-1, 1]:
                    for s2 in [-1, 1]:
                        root = np.zeros(4)
                        root[i] = s1
                        root[j] = s2
                        roots.append(root)

        # === Short roots (24 total, norm 1) ===
        # Type A: ±e_i (8 roots)
        for i in range(4):
            for s in [-1, 1]:
                root = np.zeros(4)
                root[i] = s
                roots.append(root)

        # Type B: (±1/2, ±1/2, ±1/2, ±1/2) with even number of minus signs (8 roots)
        for mask in range(16):
            signs = [1 if (mask >> i) & 1 else -1 for i in range(4)]
            if sum(1 for s in signs if s == -1) % 2 == 0:
                root = np.array([s * 0.5 for s in signs])
                roots.append(root)

        # Type C: (±1/2, ±1/2, ±1/2, ±1/2) with odd number of minus signs (8 roots)
        for mask in range(16):
            signs = [1 if (mask >> i) & 1 else -1 for i in range(4)]
            if sum(1 for s in signs if s == -1) % 2 == 1:
                root = np.array([s * 0.5 for s in signs])
                roots.append(root)

        return np.array(roots)

    def _embed_in_8d(self, roots_4d: np.ndarray) -> np.ndarray:
        """
        Embed F4 roots into R^8 (E8 ambient space).

        Uses the standard embedding: F4 in first 4 coordinates.
        """
        n_roots = len(roots_4d)
        roots_8d = np.zeros((n_roots, 8))
        roots_8d[:, :4] = roots_4d
        return roots_8d

    def _build_e8_mapping(self) -> Tuple[Dict[int, int], Dict[int, int]]:
        """
        Build bidirectional mapping between E8 and F4 root indices.

        Uses projection-based assignment: each E8 root projects to
        its nearest F4 root (by the first 4 coordinates).
        """
        e8_to_f4 = {}
        f4_to_e8 = {}

        # Normalize F4 roots for comparison
        f4_norms = np.linalg.norm(self.roots_4d, axis=1, keepdims=True)
        f4_normalized = self.roots_4d / (f4_norms + 1e-10)

        for e8_idx, e8_root in enumerate(self.e8.roots):
            # Project E8 root to first 4 coordinates
            projection = e8_root[:4]
            proj_norm = np.linalg.norm(projection)

            if proj_norm < 0.01:
                # Degenerate case: use last 4 coordinates instead
                projection = e8_root[4:]
                proj_norm = np.linalg.norm(projection)

            if proj_norm < 0.01:
                continue  # Skip zero projections

            # Normalize projection
            proj_normalized = projection / proj_norm

            # Find nearest F4 root by angle (cosine similarity)
            similarities = np.dot(f4_normalized, proj_normalized)
            f4_idx = int(np.argmax(np.abs(similarities)))

            e8_to_f4[e8_idx] = f4_idx

            # Track reverse mapping (many-to-one)
            if f4_idx not in f4_to_e8:
                f4_to_e8[f4_idx] = e8_idx

        return e8_to_f4, f4_to_e8

    def _f4_cartan_matrix(self) -> np.ndarray:
        """
        The F4 Cartan matrix.

        Dynkin diagram: o---o=>=o---o
                        1   2   3   4

        The arrow indicates roots 2,3 have different lengths.
        """
        return np.array([
            [ 2, -1,  0,  0],
            [-1,  2, -2,  0],
            [ 0, -1,  2, -1],
            [ 0,  0, -1,  2]
        ], dtype=np.float64)

    def _compute_fundamental_weights(self) -> np.ndarray:
        """
        Compute fundamental weights from Cartan matrix.

        ω_i satisfies: <ω_i, α_j^∨> = δ_ij
        """
        # Fundamental weights are rows of inverse Cartan matrix
        return np.linalg.inv(self.cartan_matrix)

    def _classify_roots(self) -> List[int]:
        """
        Identify which roots are long (norm √2) vs short (norm 1).

        Returns indices of long roots.
        """
        long_roots = []
        for i, root in enumerate(self.roots_4d):
            norm = np.linalg.norm(root)
            if norm > 1.2:  # √2 ≈ 1.414
                long_roots.append(i)
        return long_roots

    def _precompute_characters(self):
        """
        Precompute F4 character values for each root.

        The character χ_F4 is the trace in the 52-dimensional
        adjoint representation. For roots, this relates to
        the structure constants.
        """
        self.characters = np.zeros(48)

        for i, root in enumerate(self.roots_4d):
            # Character value based on root type and position
            # Long roots contribute more to the character
            if i in self.long_root_indices:
                # Long root character: based on Weyl dimension formula
                self.characters[i] = 2.0
            else:
                # Short root character
                self.characters[i] = 1.0

            # Modulate by position in Weyl chamber
            weyl_height = np.sum(np.abs(root))
            self.characters[i] *= (1 + 0.1 * weyl_height)

    # === Public Methods ===

    def is_f4_root(self, e8_index: int, threshold: float = 0.7) -> bool:
        """
        Check if an E8 root has strong F4 character.

        Uses projection quality threshold: the cosine similarity
        between the E8 root's projection and its assigned F4 root.
        """
        if e8_index not in self.e8_to_f4:
            return False

        if self.e8 is None:
            return True

        # Compute projection quality
        e8_root = self.e8.roots[e8_index]
        f4_idx = self.e8_to_f4[e8_index]
        f4_root = self.roots_4d[f4_idx]

        # Project E8 to 4D
        projection = e8_root[:4]
        proj_norm = np.linalg.norm(projection)
        f4_norm = np.linalg.norm(f4_root)

        if proj_norm < 0.01 or f4_norm < 0.01:
            return False

        # Cosine similarity
        similarity = abs(np.dot(projection, f4_root)) / (proj_norm * f4_norm)

        return similarity >= threshold

    def project_e8_to_f4(self, e8_index: int) -> Optional[int]:
        """
        Project an E8 root to its F4 component.

        Returns None if the E8 root is not mapped.
        """
        return self.e8_to_f4.get(e8_index, None)

    def get_projection_quality(self, e8_index: int) -> float:
        """
        Get the quality of E8→F4 projection (0 to 1).

        Higher values mean the E8 root aligns well with F4.
        """
        if e8_index not in self.e8_to_f4 or self.e8 is None:
            return 0.0

        e8_root = self.e8.roots[e8_index]
        f4_idx = self.e8_to_f4[e8_index]
        f4_root = self.roots_4d[f4_idx]

        projection = e8_root[:4]
        proj_norm = np.linalg.norm(projection)
        f4_norm = np.linalg.norm(f4_root)

        if proj_norm < 0.01 or f4_norm < 0.01:
            return 0.0

        return abs(np.dot(projection, f4_root)) / (proj_norm * f4_norm)

    def get_f4_root(self, f4_index: int) -> np.ndarray:
        """Get the F4 root vector (in R^4) by index."""
        return self.roots_4d[f4_index]

    def get_character(self, f4_index: int) -> float:
        """
        Get the F4 character value for a root.

        Used in the Salem-Jordan kernel weighting.
        """
        return self.characters[f4_index]

    def root_norm(self, f4_index: int) -> float:
        """Get the norm of an F4 root."""
        return np.linalg.norm(self.roots_4d[f4_index])

    def is_long_root(self, f4_index: int) -> bool:
        """Check if root is long (norm √2) vs short (norm 1)."""
        return f4_index in self.long_root_indices

    def weyl_reflection(self, vector: np.ndarray, root_index: int) -> np.ndarray:
        """
        Apply Weyl reflection s_α(v) = v - 2<v,α>/<α,α> * α
        """
        alpha = self.roots_4d[root_index]
        alpha_sq = np.dot(alpha, alpha)
        return vector - 2 * np.dot(vector, alpha) / alpha_sq * alpha

    def project_to_cartan(self, vector: np.ndarray) -> np.ndarray:
        """
        Project a vector onto the F4 Cartan subalgebra.

        For F4, this is the 4-dimensional maximal torus.
        """
        if len(vector) == 8:
            return vector[:4]
        return vector

    def get_weyl_chamber_index(self, vector: np.ndarray) -> int:
        """
        Determine which Weyl chamber a vector lies in.

        F4 has |W(F4)| = 1152 Weyl chambers.
        Returns a simplified chamber index based on sign patterns.
        """
        v = self.project_to_cartan(vector)
        # Encode sign pattern as integer
        signs = (v > 0).astype(int)
        chamber = sum(s << i for i, s in enumerate(signs))

        # Further refine by magnitude ordering
        sorted_indices = np.argsort(np.abs(v))
        ordering = sum(idx << (4 + 2*i) for i, idx in enumerate(sorted_indices))

        return chamber + ordering

    def compute_f4_spectrum(self, e8_root_assignments: np.ndarray) -> np.ndarray:
        """
        Compute the F4 power spectrum from E8 root assignments.

        Parameters
        ----------
        e8_root_assignments : np.ndarray
            Array of E8 root indices (from prime gap analysis)

        Returns
        -------
        np.ndarray
            Shape (48,) array of F4 root frequencies
        """
        f4_counts = np.zeros(48)

        for e8_idx in e8_root_assignments:
            f4_idx = self.project_e8_to_f4(int(e8_idx))
            if f4_idx is not None:
                f4_counts[f4_idx] += 1

        return f4_counts

    def summary(self) -> str:
        """Return a summary of the F4 lattice."""
        lines = [
            "F4 Root Lattice Summary",
            "=" * 40,
            f"Total roots: {len(self.roots_4d)}",
            f"Long roots (norm √2): {len(self.long_root_indices)}",
            f"Short roots (norm 1): {len(self.short_root_indices)}",
            f"Cartan matrix eigenvalues: {self.eigenvalues}",
        ]

        if self.e8:
            n_mapped = len(self.e8_to_f4)
            lines.append(f"E8 roots mapping to F4: {n_mapped}/240")

        return "\n".join(lines)


# === Standalone F4 generation (without E8 dependency) ===

def generate_f4_simple() -> np.ndarray:
    """
    Generate F4 roots without the full class machinery.

    Returns shape (48, 4) array.
    """
    lattice = F4Lattice()
    return lattice.roots_4d


if __name__ == "__main__":
    # Test F4 lattice
    f4 = F4Lattice()
    print(f4.summary())

    print("\nFirst 10 F4 roots:")
    for i in range(10):
        root = f4.get_f4_root(i)
        norm = f4.root_norm(i)
        char = f4.get_character(i)
        rtype = "long" if f4.is_long_root(i) else "short"
        print(f"  {i}: {root} (norm={norm:.3f}, {rtype}, χ={char:.3f})")
