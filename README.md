# Fluxonium Readout Simulations

This repository contains a small Dynamiqs-based scaffold for the time-dependent
readout simulations described in Sec. V and Appendix C of arXiv:2604.08515v1,
"Measurement-induced state transitions across the fluxonium qubit landscape".

The simulated master equation is

```text
d rho / dt = -i [H + H_d(t), rho] + kappa D[a] rho
```

with

```text
H = omega_r a^dag a + H_f - i g (a - a^dag) n_f
H_d(t) = -i eps(t) cos(omega_d t) (a - a^dag)
```

The code consumes the output of a fluxonium eigensolver or branch-analysis
pipeline:

- `omega_levels`: fluxonium eigenenergies relative to the ground state.
- `n_matrix`: fluxonium charge operator in the same eigenbasis.
- resonator/readout values such as `omega_r`, `g`, `kappa`, `omega_d`, and pulse
  amplitude.

All rates and frequencies are angular units in `rad/ns`; times are in `ns`.
For GHz values, use `2*pi*f_GHz`, since `1 GHz = 1/ns` in cycles/ns.

## Run the Example

```bash
MPLCONFIGDIR=.matplotlib-cache python3 -m fluxonium_readout.cli examples/example_device.json
```

The example is deliberately tiny so it runs quickly. Replace the spectra and
matrix elements in `/example_device.json` with your device values from
the branch-analysis code before running production simulations.

# Fluxonium–Resonator Readout Simulation Parameters

## Overview

This JSON configuration file defines a simplified fluxonium–resonator system used for time-domain readout simulations. The configuration is divided into three sections:

1. **Device Parameters** – Physical properties of the fluxonium and resonator.
2. **Pulse Parameters** – Characteristics of the microwave readout pulse.
3. **Simulation Parameters** – Numerical settings for the time evolution.

The system consists of a three-level fluxonium qubit coupled to a six-level microwave resonator.

---

# 1. Device Parameters

```json
"device": {
  "omega_levels": [0.0, 3.141592653589793, 10.681415022205297],
  "n_matrix": [
    [0.0, 0.08, 0.015],
    [0.08, 0.0, 0.12],
    [0.015, 0.12, 0.0]
  ],
  "omega_r": 37.69911184307752,
  "g": 0.15707963267948966,
  "kappa": 0.031415926535897934,
  "resonator_dim": 6
}
```

## Fluxonium Energy Levels

The fluxonium Hamiltonian is represented in its eigenbasis using the energy levels:

[
\omega_0 = 0
]

[
\omega_1 = \pi \ \mathrm{rad/ns}
]

[
\omega_2 = 10.6814 \ \mathrm{rad/ns}
]

corresponding to:

| Transition        | Frequency |
| ----------------- | --------- |
| (0 \rightarrow 1) | 0.5 GHz   |
| (0 \rightarrow 2) | 1.70 GHz  |

Only the three lowest fluxonium levels are retained in the simulation.

---

## Charge Operator Matrix

The matrix

[
n_{ij} = \langle i | \hat n | j \rangle
]

is supplied as

[
\hat n =
\begin{pmatrix}
0 & 0.08 & 0.015\
0.08 & 0 & 0.12\
0.015 & 0.12 & 0
\end{pmatrix}
]

These matrix elements determine how strongly the fluxonium couples to the resonator field.

Notable couplings:

| Matrix Element | Value  |           |       |
| -------------- | ------ | --------- | ----- |
| (\langle0      | \hat n | 1\rangle) | 0.08  |
| (\langle1      | \hat n | 2\rangle) | 0.12  |
| (\langle0      | \hat n | 2\rangle) | 0.015 |

---

## Resonator Frequency

The resonator frequency is

[
\omega_r = 37.6991 \ \mathrm{rad/ns}
]

or

[
f_r = \frac{\omega_r}{2\pi} = 6 \ \mathrm{GHz}
]

The resonator Hamiltonian is

[
H_r = \omega_r a^\dagger a
]

where (a) is the cavity annihilation operator.

---

## Fluxonium–Resonator Coupling

The coupling strength is

[
g = 0.15708 \ \mathrm{rad/ns}
]

corresponding to

[
\frac{g}{2\pi} = 25 \ \mathrm{MHz}
]

The interaction Hamiltonian used in the simulation is

[
H_{\text{int}}
==============

g,[-i(a-a^\dagger)],\hat n
]

This term couples the resonator quadrature to the fluxonium charge operator.

---

## Resonator Decay Rate

The cavity linewidth is

[
\kappa = 0.031416 \ \mathrm{rad/ns}
]

or

[
\frac{\kappa}{2\pi} = 5 \ \mathrm{MHz}
]

This corresponds to a cavity lifetime of approximately

[
T_{\text{cav}}
==============

\frac{1}{\kappa}
\approx 31.8 \ \mathrm{ns}
]

and enters the Lindblad master equation through

[
\kappa , \mathcal D[a].
]

---

## Resonator Hilbert Space Dimension

```json
"resonator_dim": 6
```

The resonator basis consists of

[
|0\rangle, |1\rangle, |2\rangle, |3\rangle, |4\rangle, |5\rangle.
]

The total Hilbert space dimension is therefore

[
3 \times 6 = 18.
]

---

# 2. Pulse Parameters

```json
"pulse": {
  "omega_d": 37.69911184307752,
  "amplitude": 0.2,
  "rise_time": 2.0,
  "fall_time": 2.0
}
```

## Drive Frequency

The microwave drive frequency is

[
\omega_d = \omega_r
]

which corresponds to

[
f_d = 6 \ \mathrm{GHz}.
]

The cavity is therefore driven exactly on resonance.

---

## Drive Amplitude

The pulse amplitude is

```json
"amplitude": 0.2
```

This controls the strength of the cavity excitation and therefore the number of photons generated during the readout process.

Increasing the amplitude generally:

* increases measurement signal strength,
* increases cavity population,
* may increase measurement-induced backaction.

---

## Pulse Envelope

The pulse includes smooth turn-on and turn-off ramps:

```json
"rise_time": 2.0
"fall_time": 2.0
```

giving an envelope approximately of the form

```text
      ______
     /      \
____/        \____
   2ns      2ns
```

Smooth ramps reduce spectral leakage and avoid abrupt transients.

---

# 3. Simulation Parameters

```json
"simulation": {
  "t_final": 10.0,
  "n_steps": 51,
  "measurement_efficiency": 0.4,
  "t1": 1000000.0,
  "save_states": false,
  "progress": false
}
```

## Simulation Duration

The evolution is performed over

[
t_{\text{final}} = 10 \ \mathrm{ns}.
]

---

## Time Grid

The simulation stores results at

[
N = 51
]

time points.

The timestep is therefore

[
\Delta t
========

# \frac{10}{50}

0.2 \ \mathrm{ns}.
]

---

## Measurement Efficiency

The measurement chain efficiency is

[
\eta = 0.4.
]

This means that only 40% of the photons emitted from the resonator contribute to the measured signal.

The achievable signal-to-noise ratio scales approximately as

[
\sqrt{\eta}.
]

---

## Fluxonium Relaxation Time

The qubit relaxation time is

[
T_1 = 10^6 \ \mathrm{ns}
= 1 \ \mathrm{ms}.
]

Since

[
T_1 \gg t_{\text{final}},
]

energy relaxation can be neglected during the readout window.

---

## State Storage

```json
"save_states": false
```

Only observables are stored during the simulation. Full quantum states are not retained, reducing memory usage.

---

## Progress Display

```json
"progress": false
```

Disables progress-bar output during execution.

---

# Summary

This configuration describes a three-level fluxonium qubit dispersively coupled to a six-level microwave resonator. A resonant 6 GHz readout pulse is applied to the cavity, and the resulting cavity response is simulated for 10 ns. The model includes cavity decay, finite measurement efficiency, and fluxonium–resonator coupling, while neglecting qubit relaxation over the short readout interval.
