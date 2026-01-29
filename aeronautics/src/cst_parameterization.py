"""
CST (Class Shape Transformation) Airfoil Parameterization

This module implements the CST method for airfoil representation, which is
the industry standard (Boeing, NASA) for aerodynamic shape optimization.

Reference:
    Kulfan, B. M. (2008). "Universal Parametric Geometry Representation Method"
    Journal of Aircraft, Vol. 45, No. 1, pp. 142-158.

The CST method represents airfoils using Bernstein polynomials with class functions,
allowing smooth, mathematically well-defined shapes with few parameters.
"""

import numpy as np
from scipy.special import comb
from dataclasses import dataclass
from typing import Tuple, Optional, List
import warnings


@dataclass(frozen=True)
class AirfoilConstraints:
    """Physical constraints for valid airfoil geometry."""
    min_thickness: float = 0.01      # Minimum t/c ratio
    max_thickness: float = 0.30      # Maximum t/c ratio
    max_camber: float = 0.10         # Maximum camber/c ratio
    min_le_radius: float = 0.001     # Minimum LE radius (prevents cusps)
    te_thickness: float = 0.0        # Trailing edge thickness (0 = sharp)

    def validate(self, airfoil: 'CSTAirfoil') -> Tuple[bool, List[str]]:
        """Validate airfoil against physical constraints."""
        errors = []
        coords = airfoil.get_coordinates(n_points=100)

        # Calculate thickness distribution
        upper = coords[:len(coords)//2]
        lower = coords[len(coords)//2:]
        # Approximate thickness at x=0.3 (typical max thickness location)
        thickness = airfoil.compute_max_thickness()

        if thickness < self.min_thickness:
            errors.append(f"Thickness {thickness:.4f} below minimum {self.min_thickness}")
        if thickness > self.max_thickness:
            errors.append(f"Thickness {thickness:.4f} above maximum {self.max_thickness}")

        camber = airfoil.compute_max_camber()
        if abs(camber) > self.max_camber:
            errors.append(f"Camber {camber:.4f} exceeds maximum {self.max_camber}")

        return len(errors) == 0, errors


class CSTAirfoil:
    """
    CST (Class Shape Transformation) airfoil representation.

    The airfoil shape is defined as:
        y(x) = C(x) * S(x) + x * dz_te

    Where:
        C(x) = x^N1 * (1-x)^N2  is the class function
        S(x) = sum(A_i * K_i * x^i * (1-x)^(n-i))  is the shape function

    For typical airfoils: N1=0.5, N2=1.0 (round LE, sharp TE)

    Attributes:
        weights_upper: Bernstein polynomial weights for upper surface
        weights_lower: Bernstein polynomial weights for lower surface
        n1, n2: Class function exponents (default 0.5, 1.0)
        dz_te: Trailing edge thickness offset
    """

    def __init__(
        self,
        weights_upper: np.ndarray,
        weights_lower: np.ndarray,
        n1: float = 0.5,
        n2: float = 1.0,
        dz_te: float = 0.0
    ):
        self.weights_upper = np.asarray(weights_upper, dtype=np.float64)
        self.weights_lower = np.asarray(weights_lower, dtype=np.float64)
        self.n1 = n1
        self.n2 = n2
        self.dz_te = dz_te

        # Validate inputs
        if len(self.weights_upper) != len(self.weights_lower):
            raise ValueError("Upper and lower weight arrays must have same length")
        if len(self.weights_upper) < 3:
            raise ValueError("Need at least 3 weights for meaningful representation")

    @property
    def order(self) -> int:
        """Polynomial order (number of weights - 1)."""
        return len(self.weights_upper) - 1

    @property
    def n_parameters(self) -> int:
        """Total number of shape parameters."""
        return len(self.weights_upper) + len(self.weights_lower)

    def _class_function(self, x: np.ndarray) -> np.ndarray:
        """Compute class function C(x) = x^N1 * (1-x)^N2."""
        # Handle edge cases to avoid numerical issues
        x = np.clip(x, 1e-10, 1.0 - 1e-10)
        return np.power(x, self.n1) * np.power(1.0 - x, self.n2)

    def _shape_function(self, x: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Compute shape function S(x) using Bernstein polynomials.

        S(x) = sum_{i=0}^{n} A_i * B_{i,n}(x)
        B_{i,n}(x) = C(n,i) * x^i * (1-x)^(n-i)
        """
        n = len(weights) - 1
        S = np.zeros_like(x)

        for i, A_i in enumerate(weights):
            # Bernstein basis polynomial
            K = comb(n, i, exact=True)
            B = K * np.power(x, i) * np.power(1.0 - x, n - i)
            S += A_i * B

        return S

    def _compute_surface(
        self,
        x: np.ndarray,
        weights: np.ndarray,
        sign: float = 1.0
    ) -> np.ndarray:
        """Compute y coordinates for a surface."""
        C = self._class_function(x)
        S = self._shape_function(x, weights)
        return sign * C * S + x * self.dz_te

    def get_coordinates(
        self,
        n_points: int = 200,
        cosine_spacing: bool = True
    ) -> np.ndarray:
        """
        Generate airfoil coordinates.

        Args:
            n_points: Total number of points (split between upper/lower)
            cosine_spacing: Use cosine spacing for better LE resolution

        Returns:
            Array of shape (n_points, 2) with [x, y] coordinates,
            starting from TE, going around LE, back to TE.
        """
        n_surface = n_points // 2

        if cosine_spacing:
            # Cosine spacing concentrates points near LE and TE
            beta = np.linspace(0, np.pi, n_surface)
            x = 0.5 * (1.0 - np.cos(beta))
        else:
            x = np.linspace(0, 1, n_surface)

        y_upper = self._compute_surface(x, self.weights_upper, sign=1.0)
        y_lower = self._compute_surface(x, self.weights_lower, sign=-1.0)

        # Assemble coordinates: upper surface (TE to LE) + lower surface (LE to TE)
        coords_upper = np.column_stack([x[::-1], y_upper[::-1]])
        coords_lower = np.column_stack([x[1:], y_lower[1:]])  # Skip duplicate LE point

        return np.vstack([coords_upper, coords_lower])

    def compute_max_thickness(self) -> float:
        """Compute maximum thickness-to-chord ratio."""
        x = np.linspace(0.01, 0.99, 200)
        y_upper = self._compute_surface(x, self.weights_upper, sign=1.0)
        y_lower = self._compute_surface(x, self.weights_lower, sign=-1.0)
        thickness = y_upper - y_lower
        return float(np.max(thickness))

    def compute_max_camber(self) -> float:
        """Compute maximum camber (mean line deviation from chord)."""
        x = np.linspace(0.01, 0.99, 200)
        y_upper = self._compute_surface(x, self.weights_upper, sign=1.0)
        y_lower = self._compute_surface(x, self.weights_lower, sign=-1.0)
        camber_line = (y_upper + y_lower) / 2.0
        return float(np.max(np.abs(camber_line)))

    def to_vector(self) -> np.ndarray:
        """Convert to parameter vector for ML models."""
        return np.concatenate([self.weights_upper, self.weights_lower])

    @classmethod
    def from_vector(
        cls,
        vector: np.ndarray,
        n1: float = 0.5,
        n2: float = 1.0,
        dz_te: float = 0.0
    ) -> 'CSTAirfoil':
        """Create airfoil from parameter vector."""
        n = len(vector) // 2
        return cls(
            weights_upper=vector[:n],
            weights_lower=vector[n:],
            n1=n1,
            n2=n2,
            dz_te=dz_te
        )

    @classmethod
    def from_naca4(cls, naca_code: str, n_weights: int = 6) -> 'CSTAirfoil':
        """
        Create CST airfoil approximating a NACA 4-digit airfoil.

        This is useful for validation against known airfoils.

        Args:
            naca_code: 4-digit NACA code (e.g., "2412")
            n_weights: Number of CST weights per surface
        """
        if len(naca_code) != 4 or not naca_code.isdigit():
            raise ValueError("NACA code must be 4 digits")

        m = int(naca_code[0]) / 100.0  # Max camber
        p = int(naca_code[1]) / 10.0   # Location of max camber
        t = int(naca_code[2:]) / 100.0 # Max thickness

        # Generate NACA coordinates
        x = np.linspace(0, 1, 200)

        # Thickness distribution (NACA formula)
        yt = 5 * t * (
            0.2969 * np.sqrt(x)
            - 0.1260 * x
            - 0.3516 * x**2
            + 0.2843 * x**3
            - 0.1015 * x**4  # Modified for closed TE
        )

        # Camber line
        if p > 0:
            yc = np.where(
                x < p,
                m / p**2 * (2*p*x - x**2),
                m / (1-p)**2 * ((1-2*p) + 2*p*x - x**2)
            )
        else:
            yc = np.zeros_like(x)

        y_upper = yc + yt
        y_lower = yc - yt

        # Fit CST weights using least squares
        weights_upper = cls._fit_cst_weights(x, y_upper, n_weights)
        weights_lower = cls._fit_cst_weights(x, -y_lower, n_weights)  # Note: negative for lower

        return cls(weights_upper=weights_upper, weights_lower=weights_lower)

    @staticmethod
    def _fit_cst_weights(
        x: np.ndarray,
        y: np.ndarray,
        n_weights: int,
        n1: float = 0.5,
        n2: float = 1.0
    ) -> np.ndarray:
        """Fit CST weights to given coordinates using least squares."""
        # Class function
        x_safe = np.clip(x, 1e-10, 1.0 - 1e-10)
        C = np.power(x_safe, n1) * np.power(1.0 - x_safe, n2)

        # Build Bernstein basis matrix
        n = n_weights - 1
        B = np.zeros((len(x), n_weights))
        for i in range(n_weights):
            K = comb(n, i, exact=True)
            B[:, i] = C * K * np.power(x, i) * np.power(1.0 - x, n - i)

        # Solve least squares
        weights, _, _, _ = np.linalg.lstsq(B, y, rcond=None)
        return weights


def generate_random_airfoil(
    n_weights: int = 6,
    constraints: Optional[AirfoilConstraints] = None,
    seed: Optional[int] = None,
    max_attempts: int = 100
) -> CSTAirfoil:
    """
    Generate a random physically-valid airfoil.

    Args:
        n_weights: Number of CST weights per surface
        constraints: Physical constraints to satisfy
        seed: Random seed for reproducibility
        max_attempts: Maximum generation attempts

    Returns:
        A valid CSTAirfoil instance
    """
    if constraints is None:
        constraints = AirfoilConstraints()

    rng = np.random.default_rng(seed)

    for attempt in range(max_attempts):
        # Generate weights with typical ranges
        # Upper surface: positive weights (creates upper surface)
        weights_upper = rng.uniform(0.1, 0.3, n_weights)
        # Add some variation
        weights_upper += rng.normal(0, 0.05, n_weights)
        weights_upper = np.clip(weights_upper, 0.05, 0.5)

        # Lower surface: smaller positive weights (when negated, creates lower surface)
        weights_lower = rng.uniform(0.05, 0.2, n_weights)
        weights_lower += rng.normal(0, 0.03, n_weights)
        weights_lower = np.clip(weights_lower, 0.02, 0.35)

        airfoil = CSTAirfoil(weights_upper, weights_lower)

        is_valid, errors = constraints.validate(airfoil)
        if is_valid:
            return airfoil

    warnings.warn(f"Could not generate valid airfoil in {max_attempts} attempts, "
                  f"returning last attempt. Errors: {errors}")
    return airfoil


if __name__ == "__main__":
    # Quick validation
    print("Testing CST Parameterization Module")
    print("=" * 50)

    # Test NACA 2412 conversion
    naca2412 = CSTAirfoil.from_naca4("2412")
    print(f"NACA 2412 CST representation:")
    print(f"  Upper weights: {naca2412.weights_upper}")
    print(f"  Lower weights: {naca2412.weights_lower}")
    print(f"  Max thickness: {naca2412.compute_max_thickness():.4f}")
    print(f"  Max camber: {naca2412.compute_max_camber():.4f}")

    # Validate constraints
    constraints = AirfoilConstraints()
    is_valid, errors = constraints.validate(naca2412)
    print(f"  Valid: {is_valid}, Errors: {errors}")

    # Test random generation
    print("\nRandom airfoil generation:")
    random_airfoil = generate_random_airfoil(seed=42)
    print(f"  Max thickness: {random_airfoil.compute_max_thickness():.4f}")
    print(f"  Parameters: {random_airfoil.n_parameters}")
