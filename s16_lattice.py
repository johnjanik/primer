"""
S(16) Half-Spinor Lattice — Spin(16) inside E8

The 240 E8 roots decompose under D8 = SO(16) as:
    240 = 112 (vector roots: ±e_i ± e_j) + 128 (half-spinor: (±½)^8, even neg)

The 128 half-spinor weights S+(16) form a natural sublattice of E8,
the vertices of the 8-dimensional demihypercube.

S(16) Facts:
- 128 vectors in R^8, all norm sqrt(2)
- Every vector has coordinates ±½ with an even number of minus signs
- These are exactly the Type II E8 roots
- Complementary to the 112 D8 vector roots (Type I)
- Related to the Clifford algebra Cl(16) and Bott periodicity

E8→S16 mapping: direct membership (S16 roots ARE E8 roots).
For the 112 Type I E8 roots: nearest S16 root by cosine similarity.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class S16RootData:
    """Container for S(16) analysis results"""
    s16_roots: np.ndarray          # The 128 S(16) roots in R^8
    e8_to_s16_map: Dict[int, int]  # Maps E8 root index to S16 root index
    s16_to_e8_map: Dict[int, int]  # Maps S16 root index to E8 root index


class S16Lattice:
    """
    The S(16) half-spinor lattice as a sublattice of E8.

    S16 roots are exactly the Type II E8 roots: (±½)^8 with an even
    number of minus signs. They form the weight polytope of the
    half-spin representation of Spin(16).

    The E8→S16 mapping uses direct membership for exact matches
    and cosine similarity in R^8 for the 112 Type I (vector) roots.
    """

    def __init__(self, e8_lattice=None):
        """
        Initialize S(16) lattice, optionally linking to parent E8.

        Parameters
        ----------
        e8_lattice : E8Lattice, optional
            Parent E8 lattice for cross-referencing
        """
        self.e8 = e8_lattice

        # Generate S16 roots
        self.roots_8d = self._generate_s16_roots()

        # Build mapping to E8 if available
        if e8_lattice is not None:
            self.e8_to_s16, self.s16_to_e8 = self._build_e8_mapping()
        else:
            self.e8_to_s16 = {}
            self.s16_to_e8 = {}

        # All roots are "long" (norm sqrt(2)), no short/long distinction
        self.long_root_indices = list(range(128))
        self.short_root_indices = []

        # Precompute
        self._precompute_characters()

    def _generate_s16_roots(self) -> np.ndarray:
        """
        Generate all 128 S(16) half-spinor roots in R^8.

        These are (±½)^8 with an even number of minus signs.
        """
        roots = []
        for mask in range(256):
            signs = [1 if (mask >> i) & 1 else -1 for i in range(8)]
            if sum(1 for s in signs if s == -1) % 2 == 0:
                roots.append([s * 0.5 for s in signs])

        roots = np.array(roots)
        assert len(roots) == 128, f"BUG: generated {len(roots)} S16 roots, expected 128"
        return roots

    def _build_e8_mapping(self) -> Tuple[Dict[int, int], Dict[int, int]]:
        """
        Build bidirectional mapping between E8 and S16 root indices.

        For Type II E8 roots (half-integer coords): exact membership.
        For Type I E8 roots (integer coords): nearest by cosine similarity.
        """
        e8_to_s16 = {}
        s16_to_e8 = {}

        # Precompute normalized S16 roots
        s16_norms = np.linalg.norm(self.roots_8d, axis=1, keepdims=True)
        s16_normalized = self.roots_8d / (s16_norms + 1e-10)

        for e8_idx, e8_root in enumerate(self.e8.roots):
            # Check if this is a Type II root (half-integer coords)
            is_half_int = all(abs(abs(c) - 0.5) < 1e-10 for c in e8_root)

            if is_half_int:
                # Find exact match
                diffs = np.max(np.abs(self.roots_8d - e8_root), axis=1)
                match_idx = np.argmin(diffs)
                if diffs[match_idx] < 1e-10:
                    e8_to_s16[e8_idx] = int(match_idx)
                    if int(match_idx) not in s16_to_e8:
                        s16_to_e8[int(match_idx)] = e8_idx
                    continue

            # Type I root — find nearest by cosine similarity
            e8_norm = np.linalg.norm(e8_root)
            if e8_norm < 0.01:
                continue
            e8_normalized = e8_root / e8_norm
            similarities = np.dot(s16_normalized, e8_normalized)
            s16_idx = int(np.argmax(np.abs(similarities)))
            e8_to_s16[e8_idx] = s16_idx
            if s16_idx not in s16_to_e8:
                s16_to_e8[s16_idx] = e8_idx

        return e8_to_s16, s16_to_e8

    def _precompute_characters(self):
        """
        Precompute character values for each S16 root.

        All S16 roots have norm sqrt(2). Character modulated by
        Weyl height (sum of absolute values of coordinates).
        """
        self.characters = np.zeros(128)

        for i, root in enumerate(self.roots_8d):
            weyl_height = np.sum(np.abs(root))  # always = 4.0 for half-int roots
            self.characters[i] = 2.0 * (1 + 0.1 * weyl_height)

    # === Public Methods ===

    def is_s16_root(self, e8_index: int, threshold: float = 0.7) -> bool:
        """
        Check if an E8 root has strong S16 character.

        For exact S16 roots (half-integer coords): always True.
        For Type I roots: cosine similarity threshold.
        """
        if e8_index not in self.e8_to_s16:
            return False

        if self.e8 is None:
            return True

        e8_root = self.e8.roots[e8_index]

        # Exact S16 root? (half-integer coords)
        if all(abs(abs(c) - 0.5) < 1e-10 for c in e8_root):
            return True

        # Cosine similarity check
        s16_idx = self.e8_to_s16[e8_index]
        s16_root = self.roots_8d[s16_idx]

        e8_norm = np.linalg.norm(e8_root)
        s16_norm = np.linalg.norm(s16_root)
        if e8_norm < 0.01 or s16_norm < 0.01:
            return False

        similarity = abs(np.dot(e8_root, s16_root)) / (e8_norm * s16_norm)
        return similarity >= threshold

    def project_e8_to_s16(self, e8_index: int) -> Optional[int]:
        """Project an E8 root to its S16 component. Returns None if not mapped."""
        return self.e8_to_s16.get(e8_index, None)

    def get_projection_quality(self, e8_index: int) -> float:
        """Get the quality of E8→S16 projection (0 to 1)."""
        if e8_index not in self.e8_to_s16 or self.e8 is None:
            return 0.0

        e8_root = self.e8.roots[e8_index]

        # Exact match?
        if all(abs(abs(c) - 0.5) < 1e-10 for c in e8_root):
            return 1.0

        s16_idx = self.e8_to_s16[e8_index]
        s16_root = self.roots_8d[s16_idx]

        e8_norm = np.linalg.norm(e8_root)
        s16_norm = np.linalg.norm(s16_root)
        if e8_norm < 0.01 or s16_norm < 0.01:
            return 0.0

        return abs(np.dot(e8_root, s16_root)) / (e8_norm * s16_norm)

    def get_s16_root(self, s16_index: int) -> np.ndarray:
        """Get the S16 root vector (in R^8) by index."""
        return self.roots_8d[s16_index]

    def get_character(self, s16_index: int) -> float:
        """Get the character value for a root."""
        return self.characters[s16_index]

    def root_norm(self, s16_index: int) -> float:
        """Get the norm of an S16 root (always sqrt(2))."""
        return np.linalg.norm(self.roots_8d[s16_index])

    def jordan_trace(self, s16_index: int) -> float:
        """
        Compute the trace-8 (sum of all 8 coords) for an S16 root.
        Range: [-4, +4] (each coord is ±½, sum of 8).
        """
        return np.sum(self.roots_8d[s16_index])

    def sign_parity(self, s16_index: int) -> int:
        """
        Return the number of negative coordinates (always even for S16).
        Values: 0, 2, 4, 6, or 8.
        """
        return int(np.sum(self.roots_8d[s16_index] < 0))

    def compute_s16_spectrum(self, e8_root_assignments: np.ndarray) -> np.ndarray:
        """
        Compute the S16 power spectrum from E8 root assignments.

        Returns shape (128,) array of S16 root frequencies.
        """
        s16_counts = np.zeros(128)

        for e8_idx in e8_root_assignments:
            s16_idx = self.project_e8_to_s16(int(e8_idx))
            if s16_idx is not None:
                s16_counts[s16_idx] += 1

        return s16_counts

    def summary(self) -> str:
        """Return a summary of the S16 lattice."""
        lines = [
            "S(16) Half-Spinor Lattice Summary",
            "=" * 40,
            f"Total roots: {len(self.roots_8d)}",
            f"All norm sqrt(2): {len(self.long_root_indices)}",
            f"Coords: (+-1/2)^8 with even # of minus signs",
        ]

        if self.e8:
            n_mapped = len(self.e8_to_s16)
            n_exact = sum(1 for e8_idx in self.e8_to_s16
                         if all(abs(abs(c) - 0.5) < 1e-10
                                for c in self.e8.roots[e8_idx]))
            lines.append(f"E8 roots mapping to S16: {n_mapped}/240")
            lines.append(f"  Exact S16 roots (Type II): {n_exact}")
            lines.append(f"  Approximate (Type I → nearest): {n_mapped - n_exact}")

        # Parity distribution
        parities = [self.sign_parity(i) for i in range(128)]
        from collections import Counter
        pc = Counter(parities)
        lines.append("Sign parity distribution:")
        for k in sorted(pc):
            lines.append(f"  {k} neg signs: {pc[k]} roots")

        return "\n".join(lines)


if __name__ == "__main__":
    # Test S16 lattice standalone
    s16 = S16Lattice()
    print(s16.summary())

    print("\nFirst 10 S16 roots:")
    for i in range(10):
        root = s16.get_s16_root(i)
        norm = s16.root_norm(i)
        trace = s16.jordan_trace(i)
        parity = s16.sign_parity(i)
        print(f"  {i}: {root} (norm={norm:.3f}, trace8={trace:.3f}, neg={parity})")
