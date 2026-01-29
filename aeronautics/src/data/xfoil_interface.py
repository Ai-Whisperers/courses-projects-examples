"""
XFOIL Interface for Aerodynamic Data Generation

This module provides an interface to XFOIL for generating training data.
XFOIL is a panel method solver that is fast and well-validated for subsonic flows.

For production use, XFOIL must be installed separately:
    - Linux: Available in most package managers or from MIT
    - Windows: Download from https://web.mit.edu/drela/Public/web/xfoil/

This module also provides a fallback thin airfoil theory approximation
for testing when XFOIL is not available.
"""

import subprocess
import tempfile
import os
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
from enum import Enum
import warnings
import logging

logger = logging.getLogger(__name__)


class SolverStatus(Enum):
    """Status of aerodynamic computation."""
    SUCCESS = "success"
    NOT_CONVERGED = "not_converged"
    SEPARATED = "separated"
    INVALID_INPUT = "invalid_input"
    SOLVER_ERROR = "solver_error"


@dataclass
class AeroCoefficients:
    """
    Aerodynamic coefficients from airfoil analysis.

    All coefficients are non-dimensional.
    """
    alpha: float          # Angle of attack [degrees]
    cl: float             # Lift coefficient
    cd: float             # Drag coefficient
    cm: float             # Pitching moment coefficient (about c/4)
    reynolds: float       # Reynolds number
    status: SolverStatus  # Computation status

    # Optional extended data
    cl_alpha: Optional[float] = None  # Lift curve slope [1/rad]
    cd_min: Optional[float] = None    # Minimum drag coefficient
    cl_max: Optional[float] = None    # Maximum lift coefficient
    xtr_upper: Optional[float] = None # Transition location, upper surface
    xtr_lower: Optional[float] = None # Transition location, lower surface

    def is_valid(self) -> bool:
        """Check if results are physically plausible."""
        if self.status != SolverStatus.SUCCESS:
            return False
        if self.cd <= 0:
            return False
        if abs(self.cl) > 3.0:  # Typical max for 2D airfoils
            return False
        if self.cd > 0.5:  # Very high drag indicates issues
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'alpha': self.alpha,
            'cl': self.cl,
            'cd': self.cd,
            'cm': self.cm,
            'reynolds': self.reynolds,
            'status': self.status.value,
            'cl_alpha': self.cl_alpha,
            'cd_min': self.cd_min,
            'cl_max': self.cl_max,
            'xtr_upper': self.xtr_upper,
            'xtr_lower': self.xtr_lower,
        }


@dataclass
class PressureDistribution:
    """Pressure coefficient distribution over airfoil surface."""
    x: np.ndarray         # Chordwise location [0, 1]
    cp_upper: np.ndarray  # Pressure coefficient, upper surface
    cp_lower: np.ndarray  # Pressure coefficient, lower surface
    alpha: float          # Angle of attack [degrees]


class XFOILRunner:
    """
    Interface to run XFOIL and parse results.

    This class handles:
    - Writing airfoil coordinates to temporary files
    - Generating XFOIL input scripts
    - Parsing XFOIL output
    - Error handling and validation
    """

    def __init__(
        self,
        xfoil_path: str = "xfoil",
        timeout: float = 30.0,
        max_iterations: int = 100,
        n_crit: float = 9.0  # Turbulence parameter (9 = typical wind tunnel)
    ):
        self.xfoil_path = xfoil_path
        self.timeout = timeout
        self.max_iterations = max_iterations
        self.n_crit = n_crit
        self._check_xfoil_available()

    def _check_xfoil_available(self) -> bool:
        """Check if XFOIL is available on the system."""
        try:
            result = subprocess.run(
                [self.xfoil_path],
                input=b"quit\n",
                capture_output=True,
                timeout=5.0
            )
            self.xfoil_available = True
            return True
        except (subprocess.TimeoutExpired, FileNotFoundError, PermissionError):
            self.xfoil_available = False
            logger.warning("XFOIL not found. Using analytical fallback methods.")
            return False

    def analyze(
        self,
        coordinates: np.ndarray,
        alpha: float,
        reynolds: float,
        mach: float = 0.0
    ) -> AeroCoefficients:
        """
        Analyze airfoil at given conditions.

        Args:
            coordinates: Airfoil coordinates, shape (N, 2)
            alpha: Angle of attack [degrees]
            reynolds: Reynolds number
            mach: Mach number (default 0 = incompressible)

        Returns:
            AeroCoefficients with analysis results
        """
        if not self.xfoil_available:
            return self._analytical_fallback(coordinates, alpha, reynolds)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Write coordinates
            coord_file = Path(tmpdir) / "airfoil.dat"
            self._write_coordinates(coordinates, coord_file)

            # Generate XFOIL script
            script = self._generate_script(
                coord_file, alpha, reynolds, mach, tmpdir
            )

            # Run XFOIL
            try:
                result = subprocess.run(
                    [self.xfoil_path],
                    input=script.encode(),
                    capture_output=True,
                    timeout=self.timeout,
                    cwd=tmpdir
                )

                # Parse results
                polar_file = Path(tmpdir) / "polar.txt"
                if polar_file.exists():
                    return self._parse_polar(polar_file, alpha, reynolds)
                else:
                    return AeroCoefficients(
                        alpha=alpha, cl=0.0, cd=0.0, cm=0.0,
                        reynolds=reynolds, status=SolverStatus.SOLVER_ERROR
                    )

            except subprocess.TimeoutExpired:
                logger.warning(f"XFOIL timeout at alpha={alpha}")
                return AeroCoefficients(
                    alpha=alpha, cl=0.0, cd=0.0, cm=0.0,
                    reynolds=reynolds, status=SolverStatus.NOT_CONVERGED
                )

    def _write_coordinates(self, coords: np.ndarray, filepath: Path):
        """Write coordinates in XFOIL format."""
        with open(filepath, 'w') as f:
            f.write("Airfoil\n")
            for x, y in coords:
                f.write(f" {x: .7f}  {y: .7f}\n")

    def _generate_script(
        self,
        coord_file: Path,
        alpha: float,
        reynolds: float,
        mach: float,
        tmpdir: str
    ) -> str:
        """Generate XFOIL command script."""
        script = f"""
load {coord_file}
pane
oper
visc {reynolds:.0f}
mach {mach}
iter {self.max_iterations}
vpar
n {self.n_crit}

pacc
{tmpdir}/polar.txt

alfa {alpha}

quit
"""
        return script

    def _parse_polar(
        self,
        polar_file: Path,
        alpha: float,
        reynolds: float
    ) -> AeroCoefficients:
        """Parse XFOIL polar output file."""
        try:
            with open(polar_file, 'r') as f:
                lines = f.readlines()

            # Find data lines (skip headers)
            for i, line in enumerate(lines):
                if '----' in line:
                    data_start = i + 1
                    break
            else:
                return AeroCoefficients(
                    alpha=alpha, cl=0.0, cd=0.0, cm=0.0,
                    reynolds=reynolds, status=SolverStatus.SOLVER_ERROR
                )

            if data_start >= len(lines):
                return AeroCoefficients(
                    alpha=alpha, cl=0.0, cd=0.0, cm=0.0,
                    reynolds=reynolds, status=SolverStatus.NOT_CONVERGED
                )

            # Parse last data line
            parts = lines[-1].split()
            if len(parts) >= 5:
                return AeroCoefficients(
                    alpha=float(parts[0]),
                    cl=float(parts[1]),
                    cd=float(parts[2]),
                    cm=float(parts[4]) if len(parts) > 4 else 0.0,
                    reynolds=reynolds,
                    status=SolverStatus.SUCCESS,
                    xtr_upper=float(parts[5]) if len(parts) > 5 else None,
                    xtr_lower=float(parts[6]) if len(parts) > 6 else None,
                )

            return AeroCoefficients(
                alpha=alpha, cl=0.0, cd=0.0, cm=0.0,
                reynolds=reynolds, status=SolverStatus.SOLVER_ERROR
            )

        except Exception as e:
            logger.error(f"Error parsing polar: {e}")
            return AeroCoefficients(
                alpha=alpha, cl=0.0, cd=0.0, cm=0.0,
                reynolds=reynolds, status=SolverStatus.SOLVER_ERROR
            )

    def _analytical_fallback(
        self,
        coordinates: np.ndarray,
        alpha: float,
        reynolds: float
    ) -> AeroCoefficients:
        """
        Analytical approximation when XFOIL is not available.

        Uses thin airfoil theory + flat plate drag correlation.
        This is less accurate but useful for testing the pipeline.
        """
        alpha_rad = np.radians(alpha)

        # Thin airfoil theory: Cl = 2*pi*alpha (for symmetric airfoil)
        # Add camber effect approximation
        camber = self._estimate_camber(coordinates)
        cl = 2 * np.pi * (alpha_rad + camber * 2)

        # Drag estimation (flat plate + pressure drag)
        # Blasius: Cf = 1.328 / sqrt(Re) for laminar
        # Turbulent: Cf = 0.074 / Re^0.2
        if reynolds < 5e5:
            cf = 1.328 / np.sqrt(max(reynolds, 1e4))
        else:
            cf = 0.074 / (reynolds ** 0.2)

        # Form factor for airfoil (empirical)
        thickness = self._estimate_thickness(coordinates)
        form_factor = 1 + 2 * thickness + 60 * thickness**4

        cd_friction = 2 * cf * form_factor  # Factor of 2 for both surfaces
        cd_pressure = 0.01 * abs(cl)**2  # Induced drag approximation

        cd = cd_friction + cd_pressure

        # Moment coefficient (thin airfoil theory)
        cm = -np.pi * alpha_rad / 2 - camber * np.pi

        return AeroCoefficients(
            alpha=alpha,
            cl=float(cl),
            cd=float(cd),
            cm=float(cm),
            reynolds=reynolds,
            status=SolverStatus.SUCCESS
        )

    def _estimate_camber(self, coords: np.ndarray) -> float:
        """Estimate maximum camber from coordinates."""
        n = len(coords) // 2
        upper = coords[:n]
        lower = coords[n:]

        # Approximate camber line
        # Interpolate to common x locations
        x_common = np.linspace(0.1, 0.9, 20)

        try:
            y_upper = np.interp(x_common, upper[:, 0], upper[:, 1])
            y_lower = np.interp(x_common, lower[:, 0], lower[:, 1])
            camber_line = (y_upper + y_lower) / 2
            return float(np.max(np.abs(camber_line)))
        except:
            return 0.0

    def _estimate_thickness(self, coords: np.ndarray) -> float:
        """Estimate maximum thickness from coordinates."""
        n = len(coords) // 2
        upper = coords[:n]
        lower = coords[n:]

        x_common = np.linspace(0.1, 0.9, 20)

        try:
            y_upper = np.interp(x_common, upper[:, 0], upper[:, 1])
            y_lower = np.interp(x_common, lower[:, 0], lower[:, 1])
            thickness = y_upper - y_lower
            return float(np.max(thickness))
        except:
            return 0.12  # Default assumption


class DataGenerator:
    """
    Generate training data for aerodynamic surrogate models.

    This class systematically generates aerodynamic data over
    a range of airfoils, Reynolds numbers, and angles of attack.
    """

    def __init__(
        self,
        xfoil_runner: Optional[XFOILRunner] = None,
        reynolds_range: Tuple[float, float] = (1e5, 1e7),
        alpha_range: Tuple[float, float] = (-5.0, 15.0),
        n_reynolds: int = 5,
        n_alpha: int = 21
    ):
        self.xfoil = xfoil_runner or XFOILRunner()
        self.reynolds_range = reynolds_range
        self.alpha_range = alpha_range
        self.n_reynolds = n_reynolds
        self.n_alpha = n_alpha

    def generate_polar(
        self,
        coordinates: np.ndarray,
        reynolds: float
    ) -> List[AeroCoefficients]:
        """
        Generate full polar (Cl vs alpha) for an airfoil.

        Args:
            coordinates: Airfoil coordinates
            reynolds: Reynolds number

        Returns:
            List of AeroCoefficients for each alpha
        """
        alphas = np.linspace(self.alpha_range[0], self.alpha_range[1], self.n_alpha)
        results = []

        for alpha in alphas:
            coeff = self.xfoil.analyze(coordinates, alpha, reynolds)
            if coeff.is_valid():
                results.append(coeff)

        return results

    def generate_dataset(
        self,
        airfoils: List[Tuple[np.ndarray, np.ndarray]],  # (coords, cst_params)
        progress_callback: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate full dataset for multiple airfoils.

        Args:
            airfoils: List of (coordinates, cst_parameters) tuples
            progress_callback: Optional callback for progress updates

        Returns:
            List of data dictionaries
        """
        reynolds_values = np.logspace(
            np.log10(self.reynolds_range[0]),
            np.log10(self.reynolds_range[1]),
            self.n_reynolds
        )

        dataset = []
        total = len(airfoils) * len(reynolds_values) * self.n_alpha
        current = 0

        for coords, cst_params in airfoils:
            for reynolds in reynolds_values:
                polar = self.generate_polar(coords, reynolds)

                for coeff in polar:
                    data_point = {
                        'cst_params': cst_params.tolist(),
                        'reynolds': reynolds,
                        **coeff.to_dict()
                    }
                    dataset.append(data_point)

                    current += 1
                    if progress_callback and current % 100 == 0:
                        progress_callback(current / total)

        return dataset


if __name__ == "__main__":
    # Test the module
    print("Testing XFOIL Interface Module")
    print("=" * 50)

    runner = XFOILRunner()
    print(f"XFOIL available: {runner.xfoil_available}")

    # Test with simple coordinates (NACA 0012 approximation)
    x = np.linspace(0, 1, 100)
    t = 0.12
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.126 * x - 0.3516 * x**2
                  + 0.2843 * x**3 - 0.1015 * x**4)

    upper = np.column_stack([x[::-1], yt[::-1]])
    lower = np.column_stack([x[1:], -yt[1:]])
    coords = np.vstack([upper, lower])

    result = runner.analyze(coords, alpha=5.0, reynolds=1e6)
    print(f"\nNACA 0012 at alpha=5, Re=1e6:")
    print(f"  Cl = {result.cl:.4f}")
    print(f"  Cd = {result.cd:.5f}")
    print(f"  Cm = {result.cm:.4f}")
    print(f"  Status: {result.status.value}")
    print(f"  Valid: {result.is_valid()}")
