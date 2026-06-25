from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np


@dataclass(frozen=True)
class DeviceParameters:
    """Fluxonium-resonator parameters in angular units of rad/ns."""

    omega_levels: np.ndarray
    n_matrix: np.ndarray
    n_plus_matrix: np.ndarray
    n_minus_matrix: np.ndarray
    omega_r: float
    g: float
    kappa: float
    resonator_dim: int

    @property
    def fluxonium_dim(self) -> int:
        # The fluxonium dimension is inferred from the length of the omega_levels array.
        return int(self.omega_levels.shape[0])

    def validate(self) -> None:
        # Perform basic validation checks on the parameters.
        if self.omega_levels.ndim != 1:
            raise ValueError("omega_levels must be a 1D array.")
        if self.n_matrix.shape != (self.fluxonium_dim, self.fluxonium_dim):
            raise ValueError("n_matrix must be square with size len(omega_levels).")
        if self.n_plus_matrix.shape != (self.fluxonium_dim, self.fluxonium_dim):
            raise ValueError("n_minus_matrix must be square with size len(omega_levels).")
        if self.n_minus_matrix.shape != (self.fluxonium_dim, self.fluxonium_dim):
            raise ValueError("n_plus_matrix must be square with size len(omega_levels).")
        if self.resonator_dim < 2:
            raise ValueError("resonator_dim must be at least 2.")
        if self.kappa < 0:
            raise ValueError("kappa must be non-negative.")


@dataclass(frozen=True)
class PulseParameters:
    """Readout drive pulse eps(t) cos(omega_d t)."""

    omega_d: float
    #amplitude: float
    epsilon1: float = 0.0
    epsilon2: float = 0.0
    t_switch: float = 100.0
    t_off: float = 300.0

    # rise_time: float = 0.0
    # fall_time: float = 0.0

    def validate(self) -> None:
        # Validate that the pulse parameters are physically reasonable.
        # if self.rise_time < 0 or self.fall_time < 0:
        #     raise ValueError("rise_time and fall_time must be non-negative.")
        if self.epsilon1 < 0:
            raise ValueError("epsilon1 must be non-negative.")

        if self.epsilon2 < 0:
            raise ValueError("epsilon2 must be non-negative.")

        if self.t_switch < 0:
            raise ValueError("t_switch must be non-negative.")

        if self.t_off <= self.t_switch:
            raise ValueError("t_off must be larger than t_switch.")

    # def envelope(self, t: float, total_time: float) -> float:
    #     # Compute the pulse envelope at time t, given the total pulse duration.
    #     if t < 0.0 or t > total_time:
    #         return 0.0
    #     rise = 1.0 if self.rise_time <= 0 else min(1.0, t / self.rise_time)
    #     fall_start = total_time - self.fall_time
    #     fall = 1.0
    #     if self.fall_time > 0 and t > fall_start:
    #         fall = max(0.0, (total_time - t) / self.fall_time)
    #     return self.amplitude * min(rise, fall)


@dataclass(frozen=True)
class SimulationParameters:
    """Numerical controls for a readout simulation."""

    t_final: float
    n_steps: int
    measurement_efficiency: float = 1.0
    t1: float | None = None
    save_states: bool = True
    progress: bool = False

    @property
    def tsave(self) -> np.ndarray:
        # Generate the time points at which to save the simulation results.
        return np.linspace(0.0, self.t_final, self.n_steps)


def load_json(path: str | Path) -> tuple[DeviceParameters, PulseParameters, SimulationParameters]:
    with Path(path).open("r", encoding="utf-8") as file:
        raw = json.load(file)

    device = raw["device"]
    # The pulse and simulation sections are optional, but we provide defaults for all parameters in the dataclasses.
    pulse = raw["pulse"]
    # The simulation section is optional, but we provide defaults for all parameters in the SimulationParameters dataclass.
    simulation = raw["simulation"]
    # We construct the DeviceParameters, PulseParameters, and SimulationParameters dataclasses from the loaded JSON data.

    params = DeviceParameters(
        omega_levels=np.asarray(device["omega_levels"], dtype=float),
        n_matrix=np.asarray(device["n_matrix"], dtype=complex),
        n_minus_matrix=np.asarray(device["n_minus_matrix"], dtype= complex),
        n_plus_matrix=np.asarray(device["n_plus_matrix"], dtype= complex),
        omega_r=float(device["omega_r"]),
        g=float(device["g"]),
        kappa=float(device["kappa"]),
        resonator_dim=int(device["resonator_dim"]),
    )
    params.validate()

    return (
        params,
        PulseParameters(
            omega_d=float(pulse["omega_d"]),
            epsilon1=float(pulse["epsilon1"]),
            epsilon2=float(pulse["epsilon2"]),
            t_switch=float(pulse["t_switch"]),
            t_off=float(pulse["t_off"]),
        ),
        SimulationParameters(
            t_final=float(simulation["t_final"]),
            n_steps=int(simulation["n_steps"]),
            measurement_efficiency=float(simulation.get("measurement_efficiency", 1.0)),
            t1=None if simulation.get("t1") is None else float(simulation["t1"]),
            save_states=bool(simulation.get("save_states", True)),
            progress=bool(simulation.get("progress", False)),
        ),
    )
