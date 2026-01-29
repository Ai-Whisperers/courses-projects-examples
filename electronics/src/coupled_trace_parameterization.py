"""
Coupled Trace Parameterization for PCB Crosstalk Analysis

This module implements the geometry parameterization for coupled microstrip
transmission lines, which is the foundation for crosstalk prediction.

Reference:
    Johnson & Graham (2003) "High-Speed Signal Propagation"
    Bogatin (2009) "Signal and Power Integrity - Simplified"
    Wadell (1991) "Transmission Line Design Handbook"

The coupled microstrip model characterizes two parallel traces over a ground plane,
computing even/odd mode parameters and coupling coefficients.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict, Any
import warnings


# Physical constants
C0 = 299792458.0  # Speed of light [m/s]
MU0 = 4 * np.pi * 1e-7  # Permeability of free space [H/m]
EPS0 = 8.854187817e-12  # Permittivity of free space [F/m]


@dataclass(frozen=True)
class CoupledTraceConstraints:
    """
    Manufacturing and physics constraints for coupled traces.

    These constraints represent typical PCB manufacturing limits and
    physical validity bounds for electromagnetic calculations.
    """
    # Geometry constraints [meters]
    min_spacing: float = 0.1e-3      # Minimum clearance (manufacturing)
    max_spacing: float = 5.0e-3      # Beyond this, coupling negligible
    min_width: float = 0.1e-3        # Minimum trace width
    max_width: float = 5.0e-3        # Maximum trace width
    min_height: float = 0.1e-3       # Minimum substrate height
    max_height: float = 3.2e-3       # Maximum substrate height
    min_coupled_length: float = 1e-3  # Minimum coupled length
    max_coupled_length: float = 300e-3  # Maximum coupled length
    min_thickness: float = 17e-6     # Minimum copper thickness (0.5 oz)
    max_thickness: float = 140e-6    # Maximum copper thickness (4 oz)

    # Electrical constraints
    min_dielectric_constant: float = 2.0   # Air/PTFE
    max_dielectric_constant: float = 10.0  # High-K ceramics
    min_loss_tangent: float = 0.001        # Low-loss PTFE
    max_loss_tangent: float = 0.03         # FR-4 at high frequency

    # Signal constraints
    min_frequency: float = 1e3       # 1 kHz
    max_frequency: float = 10e9      # 10 GHz
    min_rise_time: float = 50e-12    # 50 ps
    max_rise_time: float = 100e-9    # 100 ns
    min_termination: float = 25.0    # Minimum termination impedance [ohm]
    max_termination: float = 150.0   # Maximum termination impedance [ohm]

    # Coupling coefficient limits
    max_coupling_coefficient: float = 0.5  # Unrealistic above this

    def validate(self, traces: 'CoupledMicrostrips') -> Tuple[bool, List[str]]:
        """
        Validate coupled traces against physical constraints.

        Args:
            traces: CoupledMicrostrips instance to validate

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []

        # Geometry validation
        if traces.width < self.min_width:
            errors.append(f"Width {traces.width*1e3:.3f}mm below minimum {self.min_width*1e3:.3f}mm")
        if traces.width > self.max_width:
            errors.append(f"Width {traces.width*1e3:.3f}mm above maximum {self.max_width*1e3:.3f}mm")

        if traces.spacing < self.min_spacing:
            errors.append(f"Spacing {traces.spacing*1e3:.3f}mm below minimum {self.min_spacing*1e3:.3f}mm")
        if traces.spacing > self.max_spacing:
            errors.append(f"Spacing {traces.spacing*1e3:.3f}mm above maximum {self.max_spacing*1e3:.3f}mm")

        if traces.height < self.min_height:
            errors.append(f"Height {traces.height*1e3:.3f}mm below minimum {self.min_height*1e3:.3f}mm")
        if traces.height > self.max_height:
            errors.append(f"Height {traces.height*1e3:.3f}mm above maximum {self.max_height*1e3:.3f}mm")

        if traces.coupled_length < self.min_coupled_length:
            errors.append(f"Coupled length {traces.coupled_length*1e3:.1f}mm below minimum")
        if traces.coupled_length > self.max_coupled_length:
            errors.append(f"Coupled length {traces.coupled_length*1e3:.1f}mm above maximum")

        if traces.thickness < self.min_thickness:
            errors.append(f"Copper thickness {traces.thickness*1e6:.1f}um below minimum")
        if traces.thickness > self.max_thickness:
            errors.append(f"Copper thickness {traces.thickness*1e6:.1f}um above maximum")

        # Electrical validation
        if traces.dielectric_constant < self.min_dielectric_constant:
            errors.append(f"Dielectric constant {traces.dielectric_constant:.2f} below minimum")
        if traces.dielectric_constant > self.max_dielectric_constant:
            errors.append(f"Dielectric constant {traces.dielectric_constant:.2f} above maximum")

        if traces.loss_tangent < self.min_loss_tangent:
            errors.append(f"Loss tangent {traces.loss_tangent:.4f} below minimum")
        if traces.loss_tangent > self.max_loss_tangent:
            errors.append(f"Loss tangent {traces.loss_tangent:.4f} above maximum")

        # Signal validation
        if traces.frequency < self.min_frequency:
            errors.append(f"Frequency {traces.frequency/1e6:.3f}MHz below minimum")
        if traces.frequency > self.max_frequency:
            errors.append(f"Frequency {traces.frequency/1e9:.3f}GHz above maximum")

        if traces.rise_time < self.min_rise_time:
            errors.append(f"Rise time {traces.rise_time*1e12:.1f}ps below minimum")
        if traces.rise_time > self.max_rise_time:
            errors.append(f"Rise time {traces.rise_time*1e9:.1f}ns above maximum")

        if traces.termination_impedance < self.min_termination:
            errors.append(f"Termination {traces.termination_impedance:.1f}ohm below minimum")
        if traces.termination_impedance > self.max_termination:
            errors.append(f"Termination {traces.termination_impedance:.1f}ohm above maximum")

        # Coupling coefficient validation
        kb, kf = traces.compute_coupling_coefficients()
        if abs(kb) > self.max_coupling_coefficient:
            errors.append(f"Backward coupling |Kb|={abs(kb):.3f} exceeds limit")
        if abs(kf) > self.max_coupling_coefficient:
            errors.append(f"Forward coupling |Kf|={abs(kf):.3f} exceeds limit")

        return len(errors) == 0, errors


class CoupledMicrostrips:
    """
    Coupled microstrip transmission line pair.

    Models two parallel traces over a ground plane, computing:
    - Even/odd mode impedances
    - Differential/common mode impedances
    - Mutual capacitance and inductance
    - Coupling coefficients (Kb, Kf)

    The 10 parameters characterize the complete geometry and signal:
    1. width (w): Trace width [m]
    2. spacing (s): Edge-to-edge separation [m]
    3. height (h): Substrate height [m]
    4. coupled_length (L): Length of coupled region [m]
    5. thickness (t): Copper thickness [m]
    6. dielectric_constant (er): Substrate relative permittivity
    7. loss_tangent (tan_d): Substrate loss tangent
    8. frequency (f): Signal frequency [Hz]
    9. rise_time (Tr): Signal rise time [s]
    10. termination_impedance (Zt): Load impedance [ohm]

    Attributes:
        width: Trace width [m]
        spacing: Edge-to-edge trace spacing [m]
        height: Substrate thickness above ground plane [m]
        coupled_length: Length of parallel coupled region [m]
        thickness: Copper thickness [m]
        dielectric_constant: Substrate relative permittivity
        loss_tangent: Substrate dielectric loss tangent
        frequency: Operating frequency [Hz]
        rise_time: Signal rise time (10-90%) [s]
        termination_impedance: Termination impedance [ohm]
    """

    def __init__(
        self,
        width: float,
        spacing: float,
        height: float,
        coupled_length: float,
        thickness: float = 35e-6,  # 1 oz copper default
        dielectric_constant: float = 4.3,  # FR-4 typical
        loss_tangent: float = 0.02,  # FR-4 typical
        frequency: float = 1e9,  # 1 GHz default
        rise_time: float = 1e-9,  # 1 ns default
        termination_impedance: float = 50.0  # 50 ohm default
    ):
        self.width = width
        self.spacing = spacing
        self.height = height
        self.coupled_length = coupled_length
        self.thickness = thickness
        self.dielectric_constant = dielectric_constant
        self.loss_tangent = loss_tangent
        self.frequency = frequency
        self.rise_time = rise_time
        self.termination_impedance = termination_impedance

        # Validate basic physics
        if width <= 0 or spacing <= 0 or height <= 0:
            raise ValueError("Width, spacing, and height must be positive")
        if coupled_length <= 0:
            raise ValueError("Coupled length must be positive")
        if dielectric_constant < 1.0:
            raise ValueError("Dielectric constant must be >= 1.0")

    @property
    def n_parameters(self) -> int:
        """Total number of parameters."""
        return 10

    def _compute_effective_dielectric_constant(self) -> float:
        """
        Compute effective dielectric constant for microstrip.

        Uses the Hammerstad-Jensen formula for effective permittivity.
        """
        er = self.dielectric_constant
        w_h = self.width / self.height

        # Hammerstad-Jensen formula
        a = 1 + (1/49) * np.log((w_h**4 + (w_h/52)**2) / (w_h**4 + 0.432)) + \
            (1/18.7) * np.log(1 + (w_h/18.1)**3)
        b = 0.564 * ((er - 0.9) / (er + 3))**0.053

        eff_er = (er + 1) / 2 + (er - 1) / 2 * (1 + 10 / w_h)**(-a * b)

        return eff_er

    def _compute_single_microstrip_impedance(self) -> float:
        """
        Compute characteristic impedance of a single microstrip line.

        Uses Wheeler's synthesis formula with Hammerstad corrections.
        """
        w = self.width
        h = self.height
        t = self.thickness
        er = self.dielectric_constant

        # Effective width accounting for thickness
        delta_w = (t / np.pi) * np.log(1 + 4 * np.exp(1) * h / (t * np.cosh(np.sqrt(6.517 * w / h))**2))
        w_eff = w + delta_w

        w_h = w_eff / h
        eff_er = self._compute_effective_dielectric_constant()

        # Wheeler's formula
        if w_h <= 1:
            F = 6 + (2 * np.pi - 6) * np.exp(-(30.666 / w_h)**0.7528)
            Z0 = (60 / np.sqrt(eff_er)) * np.log(F / w_h + np.sqrt(1 + (2/w_h)**2))
        else:
            Z0 = (120 * np.pi) / (np.sqrt(eff_er) * (w_h + 1.393 + 0.667 * np.log(w_h + 1.444)))

        return Z0

    def compute_even_mode_impedance(self) -> float:
        """
        Compute even-mode characteristic impedance.

        In even mode, both traces are driven with the same polarity,
        and the coupling enhances the capacitance to ground.

        Returns:
            Even-mode impedance Z0e [ohm]
        """
        Z0 = self._compute_single_microstrip_impedance()
        er = self.dielectric_constant
        eff_er = self._compute_effective_dielectric_constant()

        # Normalized dimensions
        w = self.width
        h = self.height
        s = self.spacing

        w_h = w / h
        s_h = s / h

        # Garg-Bahl formula for even mode
        # Coupling factor depends on spacing
        g = s / (s + 2 * w)  # Gap ratio

        # Even mode capacitance increase factor
        q_even = 0.5 * np.exp(-0.7 * s_h) * (1 - np.exp(-2 * w_h))

        # Even mode effective dielectric constant
        eff_er_e = eff_er * (1 + q_even * (er - 1) / eff_er)

        # Even mode impedance (higher than single line)
        Z0e = Z0 * np.sqrt(eff_er / eff_er_e) * (1 + (0.2 / (1 + 0.5 * s_h**2)))

        return Z0e

    def compute_odd_mode_impedance(self) -> float:
        """
        Compute odd-mode characteristic impedance.

        In odd mode, traces are driven with opposite polarity,
        creating a virtual ground between them.

        Returns:
            Odd-mode impedance Z0o [ohm]
        """
        Z0 = self._compute_single_microstrip_impedance()
        er = self.dielectric_constant
        eff_er = self._compute_effective_dielectric_constant()

        w = self.width
        h = self.height
        s = self.spacing

        w_h = w / h
        s_h = s / h

        # Odd mode capacitance increase factor (stronger coupling)
        q_odd = 0.5 * np.exp(-0.6 * s_h) * (1 + np.exp(-1.5 * w_h))

        # Odd mode effective dielectric constant
        eff_er_o = eff_er * (1 + q_odd * (er - 1) / eff_er)

        # Odd mode impedance (lower than single line)
        Z0o = Z0 * np.sqrt(eff_er / eff_er_o) * (1 - (0.3 / (1 + 0.4 * s_h**2)))

        return Z0o

    def compute_differential_impedance(self) -> float:
        """
        Compute differential impedance for differential signaling.

        Zdiff = 2 * Z0odd (for loosely coupled lines)

        Returns:
            Differential impedance [ohm]
        """
        Z0o = self.compute_odd_mode_impedance()
        return 2 * Z0o

    def compute_common_mode_impedance(self) -> float:
        """
        Compute common-mode impedance.

        Zcommon = Z0even / 2

        Returns:
            Common-mode impedance [ohm]
        """
        Z0e = self.compute_even_mode_impedance()
        return Z0e / 2

    def compute_per_unit_length_parameters(self) -> Dict[str, float]:
        """
        Compute per-unit-length L and C parameters.

        Returns:
            Dictionary with L0, C0, Lm, Cm (per unit length parameters)
        """
        Z0e = self.compute_even_mode_impedance()
        Z0o = self.compute_odd_mode_impedance()
        eff_er = self._compute_effective_dielectric_constant()

        # Phase velocity
        v = C0 / np.sqrt(eff_er)

        # Self parameters from single line
        L0 = Z0e * Z0o / v  # Geometric mean approximation
        C0 = 1 / (v * np.sqrt(Z0e * Z0o))

        # Mutual parameters from mode impedances
        # Z0e = sqrt((L0 + Lm) / (C0 - Cm))
        # Z0o = sqrt((L0 - Lm) / (C0 + Cm))

        # Solving for Lm and Cm
        Lm = (Z0e - Z0o) / (2 * v)  # Mutual inductance per unit length
        Cm = (1/Z0o - 1/Z0e) / (2 * v)  # Mutual capacitance per unit length

        return {
            'L0': L0,  # [H/m]
            'C0': C0,  # [F/m]
            'Lm': Lm,  # [H/m]
            'Cm': Cm,  # [F/m]
        }

    def compute_mutual_capacitance(self) -> float:
        """
        Compute mutual capacitance per unit length.

        Returns:
            Mutual capacitance [F/m]
        """
        params = self.compute_per_unit_length_parameters()
        return params['Cm']

    def compute_mutual_inductance(self) -> float:
        """
        Compute mutual inductance per unit length.

        Returns:
            Mutual inductance [H/m]
        """
        params = self.compute_per_unit_length_parameters()
        return params['Lm']

    def compute_coupling_coefficients(self) -> Tuple[float, float]:
        """
        Compute backward (Kb) and forward (Kf) coupling coefficients.

        Kb = (1/4) * (Cm/C0 + Lm/L0)  # Backward (NEXT)
        Kf = (1/2) * (Cm/C0 - Lm/L0)  # Forward (FEXT)

        For homogeneous medium: Kf = 0 (capacitive = inductive coupling)
        For microstrip (inhomogeneous): Kf != 0

        Returns:
            Tuple of (Kb, Kf)
        """
        params = self.compute_per_unit_length_parameters()
        L0 = params['L0']
        C0 = params['C0']
        Lm = params['Lm']
        Cm = params['Cm']

        # Coupling coefficients
        Kb = 0.25 * (Cm / C0 + Lm / L0)
        Kf = 0.5 * (Cm / C0 - Lm / L0)

        return Kb, Kf

    def compute_propagation_delay(self) -> float:
        """
        Compute propagation delay per unit length.

        Returns:
            Propagation delay [s/m]
        """
        eff_er = self._compute_effective_dielectric_constant()
        return np.sqrt(eff_er) / C0

    def compute_wavelength(self) -> float:
        """
        Compute wavelength at operating frequency.

        Returns:
            Wavelength [m]
        """
        eff_er = self._compute_effective_dielectric_constant()
        return C0 / (self.frequency * np.sqrt(eff_er))

    def compute_electrical_length(self) -> float:
        """
        Compute electrical length in wavelengths.

        Returns:
            Electrical length L/lambda
        """
        wavelength = self.compute_wavelength()
        return self.coupled_length / wavelength

    def to_vector(self) -> np.ndarray:
        """
        Convert parameters to vector for ML models.

        Order: [w, s, h, L, t, er, tan_d, f, Tr, Zt]
        All values are in SI units.

        Returns:
            numpy array of shape (10,)
        """
        return np.array([
            self.width,
            self.spacing,
            self.height,
            self.coupled_length,
            self.thickness,
            self.dielectric_constant,
            self.loss_tangent,
            self.frequency,
            self.rise_time,
            self.termination_impedance,
        ], dtype=np.float64)

    @classmethod
    def from_vector(
        cls,
        vector: np.ndarray
    ) -> 'CoupledMicrostrips':
        """
        Create instance from parameter vector.

        Args:
            vector: Array of shape (10,) with parameters in SI units

        Returns:
            CoupledMicrostrips instance
        """
        if len(vector) != 10:
            raise ValueError(f"Expected 10 parameters, got {len(vector)}")

        return cls(
            width=vector[0],
            spacing=vector[1],
            height=vector[2],
            coupled_length=vector[3],
            thickness=vector[4],
            dielectric_constant=vector[5],
            loss_tangent=vector[6],
            frequency=vector[7],
            rise_time=vector[8],
            termination_impedance=vector[9],
        )

    @classmethod
    def from_standard(cls, standard: str) -> 'CoupledMicrostrips':
        """
        Create coupled traces matching common interface standards.

        Args:
            standard: One of "USB2_DIFF", "PCIE_GEN3", "DDR4",
                     "100BASE_TX", "LVDS", "LOOSE_COUPLED", "TIGHT_COUPLED"

        Returns:
            CoupledMicrostrips configured for the standard
        """
        standards = {
            "USB2_DIFF": {
                # USB 2.0 Hi-Speed: 90 ohm differential
                "width": 0.2e-3,
                "spacing": 0.15e-3,
                "height": 0.2e-3,
                "coupled_length": 100e-3,
                "thickness": 35e-6,
                "dielectric_constant": 4.3,
                "loss_tangent": 0.02,
                "frequency": 480e6,
                "rise_time": 500e-12,
                "termination_impedance": 45.0,
            },
            "PCIE_GEN3": {
                # PCIe Gen3: 85 ohm differential, 8 GT/s
                "width": 0.15e-3,
                "spacing": 0.18e-3,
                "height": 0.15e-3,
                "coupled_length": 200e-3,
                "thickness": 35e-6,
                "dielectric_constant": 3.5,
                "loss_tangent": 0.015,
                "frequency": 4e9,
                "rise_time": 50e-12,
                "termination_impedance": 42.5,
            },
            "DDR4": {
                # DDR4-2400: SSTL signaling
                "width": 0.1e-3,
                "spacing": 0.15e-3,
                "height": 0.1e-3,
                "coupled_length": 50e-3,
                "thickness": 35e-6,
                "dielectric_constant": 4.3,
                "loss_tangent": 0.02,
                "frequency": 1.2e9,
                "rise_time": 300e-12,
                "termination_impedance": 40.0,
            },
            "100BASE_TX": {
                # 100 Mbps Ethernet: 100 ohm differential
                "width": 0.25e-3,
                "spacing": 0.2e-3,
                "height": 0.25e-3,
                "coupled_length": 150e-3,
                "thickness": 35e-6,
                "dielectric_constant": 4.3,
                "loss_tangent": 0.02,
                "frequency": 125e6,
                "rise_time": 4e-9,
                "termination_impedance": 50.0,
            },
            "LVDS": {
                # Low Voltage Differential Signaling: 100 ohm differential
                "width": 0.15e-3,
                "spacing": 0.15e-3,
                "height": 0.15e-3,
                "coupled_length": 100e-3,
                "thickness": 35e-6,
                "dielectric_constant": 4.0,
                "loss_tangent": 0.018,
                "frequency": 655e6,
                "rise_time": 500e-12,
                "termination_impedance": 50.0,
            },
            "LOOSE_COUPLED": {
                # Reference: Loosely coupled (s = 3w)
                "width": 0.2e-3,
                "spacing": 0.6e-3,  # s = 3w
                "height": 0.2e-3,
                "coupled_length": 50e-3,
                "thickness": 35e-6,
                "dielectric_constant": 4.3,
                "loss_tangent": 0.02,
                "frequency": 1e9,
                "rise_time": 500e-12,
                "termination_impedance": 50.0,
            },
            "TIGHT_COUPLED": {
                # Reference: Tightly coupled (s = w)
                "width": 0.2e-3,
                "spacing": 0.2e-3,  # s = w
                "height": 0.2e-3,
                "coupled_length": 50e-3,
                "thickness": 35e-6,
                "dielectric_constant": 4.3,
                "loss_tangent": 0.02,
                "frequency": 1e9,
                "rise_time": 500e-12,
                "termination_impedance": 50.0,
            },
        }

        if standard not in standards:
            available = ", ".join(standards.keys())
            raise ValueError(f"Unknown standard '{standard}'. Available: {available}")

        return cls(**standards[standard])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'width': self.width,
            'spacing': self.spacing,
            'height': self.height,
            'coupled_length': self.coupled_length,
            'thickness': self.thickness,
            'dielectric_constant': self.dielectric_constant,
            'loss_tangent': self.loss_tangent,
            'frequency': self.frequency,
            'rise_time': self.rise_time,
            'termination_impedance': self.termination_impedance,
        }

    def summary(self) -> str:
        """Generate human-readable summary of trace parameters."""
        Z0e = self.compute_even_mode_impedance()
        Z0o = self.compute_odd_mode_impedance()
        Zdiff = self.compute_differential_impedance()
        Kb, Kf = self.compute_coupling_coefficients()

        return f"""Coupled Microstrip Parameters:
  Geometry:
    Width (w):       {self.width*1e3:.3f} mm
    Spacing (s):     {self.spacing*1e3:.3f} mm
    Height (h):      {self.height*1e3:.3f} mm
    Length (L):      {self.coupled_length*1e3:.1f} mm
    Thickness (t):   {self.thickness*1e6:.1f} um
  Material:
    Dielectric (er): {self.dielectric_constant:.2f}
    Loss tangent:    {self.loss_tangent:.4f}
  Signal:
    Frequency:       {self.frequency/1e9:.3f} GHz
    Rise time:       {self.rise_time*1e12:.1f} ps
    Termination:     {self.termination_impedance:.1f} ohm
  Computed:
    Z0 even:         {Z0e:.1f} ohm
    Z0 odd:          {Z0o:.1f} ohm
    Z differential:  {Zdiff:.1f} ohm
    Kb (backward):   {Kb:.4f}
    Kf (forward):    {Kf:.4f}
    L/lambda:        {self.compute_electrical_length():.3f}"""


def generate_random_traces(
    constraints: Optional[CoupledTraceConstraints] = None,
    seed: Optional[int] = None,
    max_attempts: int = 100
) -> CoupledMicrostrips:
    """
    Generate random physically-valid coupled traces.

    Args:
        constraints: Physical constraints to satisfy
        seed: Random seed for reproducibility
        max_attempts: Maximum generation attempts

    Returns:
        A valid CoupledMicrostrips instance
    """
    if constraints is None:
        constraints = CoupledTraceConstraints()

    rng = np.random.default_rng(seed)

    for attempt in range(max_attempts):
        # Generate random parameters within ranges
        width = rng.uniform(constraints.min_width, constraints.max_width)
        spacing = rng.uniform(constraints.min_spacing, constraints.max_spacing)
        height = rng.uniform(constraints.min_height, constraints.max_height)
        coupled_length = rng.uniform(constraints.min_coupled_length, constraints.max_coupled_length)
        thickness = rng.uniform(constraints.min_thickness, constraints.max_thickness)

        # Log-uniform for frequency
        log_f_min = np.log10(constraints.min_frequency)
        log_f_max = np.log10(constraints.max_frequency)
        frequency = 10 ** rng.uniform(log_f_min, log_f_max)

        # Log-uniform for rise time
        log_tr_min = np.log10(constraints.min_rise_time)
        log_tr_max = np.log10(constraints.max_rise_time)
        rise_time = 10 ** rng.uniform(log_tr_min, log_tr_max)

        dielectric_constant = rng.uniform(
            constraints.min_dielectric_constant,
            constraints.max_dielectric_constant
        )
        loss_tangent = rng.uniform(
            constraints.min_loss_tangent,
            constraints.max_loss_tangent
        )
        termination_impedance = rng.uniform(
            constraints.min_termination,
            constraints.max_termination
        )

        traces = CoupledMicrostrips(
            width=width,
            spacing=spacing,
            height=height,
            coupled_length=coupled_length,
            thickness=thickness,
            dielectric_constant=dielectric_constant,
            loss_tangent=loss_tangent,
            frequency=frequency,
            rise_time=rise_time,
            termination_impedance=termination_impedance,
        )

        is_valid, errors = constraints.validate(traces)
        if is_valid:
            return traces

    warnings.warn(f"Could not generate valid traces in {max_attempts} attempts, "
                  f"returning last attempt. Errors: {errors}")
    return traces


if __name__ == "__main__":
    # Quick validation
    print("Testing Coupled Trace Parameterization Module")
    print("=" * 60)

    # Test standard configurations
    for std in ["USB2_DIFF", "PCIE_GEN3", "LOOSE_COUPLED", "TIGHT_COUPLED"]:
        traces = CoupledMicrostrips.from_standard(std)
        print(f"\n{std}:")
        print(f"  Zdiff = {traces.compute_differential_impedance():.1f} ohm")
        Kb, Kf = traces.compute_coupling_coefficients()
        print(f"  Kb = {Kb:.4f}, Kf = {Kf:.4f}")

        # Validate constraints
        constraints = CoupledTraceConstraints()
        is_valid, errors = constraints.validate(traces)
        print(f"  Valid: {is_valid}")
        if errors:
            for err in errors:
                print(f"    - {err}")

    # Test vector conversion
    print("\nVector conversion test:")
    traces = CoupledMicrostrips.from_standard("PCIE_GEN3")
    vec = traces.to_vector()
    print(f"  Vector shape: {vec.shape}")
    traces_recovered = CoupledMicrostrips.from_vector(vec)
    print(f"  Zdiff original: {traces.compute_differential_impedance():.1f} ohm")
    print(f"  Zdiff recovered: {traces_recovered.compute_differential_impedance():.1f} ohm")

    # Test random generation
    print("\nRandom trace generation:")
    random_traces = generate_random_traces(seed=42)
    print(random_traces.summary())
