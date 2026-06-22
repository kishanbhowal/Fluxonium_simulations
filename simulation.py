from __future__ import annotations

from dataclasses import dataclass
from unittest import result

import jax.numpy as jnp
import numpy as np
from scipy.special import erfc
import dynamiqs as dq
from dynamiqs.method import Dopri8, Tsit5, Dopri5
import time

from torch import ops

from hamiltonian import (
    basis_state,
    computational_projector,
    fluxonium_projector,
    jump_operators,
    operators,
    readout_hamiltonian,
    readout_hamiltonian_rwa,
)
from parameters import DeviceParameters, PulseParameters, SimulationParameters


@dataclass(frozen=True)
class ReadoutResult:
    tsave: np.ndarray # Time points at which the simulation results are saved.
    resonator_field: np.ndarray # Expectation value of the resonator field (annihilation operator) as a function of time.
    resonator_population: np.ndarray # Expectation value of the resonator population (number operator) as a function of time.
    fluxonium_populations: np.ndarray # Expectation values of the fluxonium populations (number operators) as a function of time.
    leakage: np.ndarray # Expectation value of the leakage probability as a function of time.


@dataclass(frozen=True)
class ReadoutPairResult:
    ground: ReadoutResult # Readout result for the ground state of the fluxonium.
    excited: ReadoutResult # Readout result for the excited state of the fluxonium.
    snr: np.ndarray # Signal-to-noise ratio as a function of time.
    assignment_error: np.ndarray # Assignment error as a function of time.


def run_readout_trajectory(
    params: DeviceParameters,
    pulse: PulseParameters,
    sim: SimulationParameters,
    initial_fluxonium_state: int,
) -> ReadoutResult:
    
    # Set up the Hamiltonian, jump operators, and initial state for the simulation, based on the provided parameters and pulse. We then run the master equation solver to simulate the time evolution of the system, and extract the relevant expectation values to construct the ReadoutResult.
    ops = operators(params) # Get the relevant operators for the system, including the annihilation operator, number operator, and Hamiltonian terms.
    hamiltonian = readout_hamiltonian_rwa(params, pulse)# Construct the time-dependent Hamiltonian for the readout, based on the device parameters, pulse parameters, and total simulation time.
    #ops = operators(params)
    jumps = jump_operators(params,ops["a"])# Get the list of jump operators for the system, which represent the dissipation processes. In this case, we only include the resonator decay as a jump operator.
    psi0 = basis_state(params, initial_fluxonium_state)# Construct the initial state for the simulation, which is a basis state corresponding to the specified initial fluxonium state and the resonator in the vacuum state. The state is represented as a QArray.
    rho0 = psi0 @ psi0.dag()# Construct the initial density matrix for the simulation, which is given by the outer product of the initial state with itself. This represents a pure state in the density matrix formalism.

    projectors = [fluxonium_projector(params, idx) for idx in range(params.fluxonium_dim)] # Construct a list of projectors onto each fluxonium state, which will be used to extract the populations of the fluxonium states from the simulation results. Each projector is represented as a QArray, constructed as the tensor product of the projector onto the specified fluxonium state and the identity operator on the resonator subsystem.
    exp_ops = [ops["a"], ops["n_photon"], *projectors, computational_projector(params)] # Construct the list of operators for which we want to compute expectation values during the simulation. This includes the annihilation operator for the resonator, the number operator for the resonator, the projectors onto each fluxonium state, and a projector onto the computational subspace. Each operator is represented as a QArray.

    dt = sim.tsave[1] - sim.tsave[0] # Compute the time step between the saved time points, which will be used to determine the time span for each segment of the simulation. This is important for ensuring that the simulation is run with the correct time resolution and that the results are saved at the desired time points.
    chunk_time = 100.0
    chunk = max(1, int(round(chunk_time / dt))) # Determine the number of time steps to include in each chunk of the simulation, based on the specified chunk time and the time step. This allows us to run the simulation in segments, which can help manage memory usage and improve performance for long simulations.
    all_expects = []
    
    rho = rho0
    for i in range(0, len(sim.tsave)-1, chunk):
        stop = min(i+chunk + 1, len(sim.tsave))
        tspan = sim.tsave[i:stop] # Define the time span for the current chunk of the simulation, which includes the time points from the current index to the end of the chunk or the end of the total time span, whichever comes first. This allows us to run the simulation in segments and save the results at the specified time points.
        
        start_time = time.time()

        result = dq.mesolve(
            hamiltonian,
            jumps,


            rho,
            tspan,
            exp_ops=exp_ops,
            method= Tsit5(rtol=1e-4, atol=1e-4, max_steps=10000000),
            options=dq.Options(save_states= True, progress_meter=sim.progress, t0 = tspan[0]),
        )
        print(result)
        print(result.infos)
        print(f"Chunk {i//chunk + 1} / {(len(sim.tsave)-1 + chunk - 1) // chunk}, Time: {time.time() - start_time:.2f} seconds")
        if i == 0:
            all_expects.append(np.asarray(result.expects))
        else:
            all_expects.append(np.asarray(result.expects)[:, 1:])
        rho = result.final_state # Update the initial state for the next chunk of the simulation to be the final state from the current chunk. This allows us to run the simulation in segments while maintaining continuity in the state evolution across chunks.


    # result = dq.mesolve( # Run the master equation solver to simulate the time evolution of the system under the specified Hamiltonian and jump operators, starting from the initial state. We save the expectation values of the specified operators at the time points given by sim.tsave, and we can also save the full state trajectories if sim.save_states is True. The progress of the simulation can be displayed if sim.progress is True.
    #     hamiltonian,
    #     jumps,
    #     rho0,
    #     sim.tsave,
    #     exp_ops=exp_ops,
    #     method= Dopri8(rtol=1e-4, atol=1e-4, max_steps=10000000),
    #     options=dq.Options(save_states=sim.save_states, progress_meter=sim.progress),
    # )
    #expects = np.asarray(result.expects) # Extract the expectation values from the simulation results, and convert them to NumPy arrays for further processing. The expects array will have shape (number of time points, number of operators), where each column corresponds to the expectation value of a specific operator at each time point.
    expects = np.concatenate(all_expects, axis=1)
    resonator_field = expects[0] # The first column of the expects array corresponds to the expectation value of the resonator field (annihilation operator) as a function of time.
    resonator_population = np.maximum(expects[1].real, 0.0) # The second column of the expects array corresponds to the expectation value of the resonator population (number operator) as a function of time. We take the real part and ensure that it is non-negative, since the population cannot be negative.
    fluxonium_populations = np.clip(expects[2 : 2 + params.fluxonium_dim].real, 0.0, 1.0) # The next columns of the expects array correspond to the expectation values of the fluxonium populations (number operators) for each fluxonium state. We take the real part and clip the values to be between 0 and 1, since these represent probabilities.
    computational_population = np.clip(expects[2 + params.fluxonium_dim].real, 0.0, 1.0) # The last column of the expects array corresponds to the expectation value of the projector onto the computational subspace, which represents the population in the computational states. We take the real part and clip it to be between 0 and 1, since this represents a probability.
    leakage = np.clip(1.0 - computational_population, 0.0, 1.0) # The leakage is given by one minus the population in the computational subspace, since any population outside of the computational subspace represents leakage. We clip the value to be between 0 and 1, since this represents a probability.

    return ReadoutResult(
        tsave=np.asarray(sim.tsave), # Time points at which the simulation results are saved, converted to a NumPy array.
        resonator_field=np.asarray(resonator_field), # Expectation value of the resonator field (annihilation operator) as a function of time, converted to a NumPy array.
        resonator_population=np.asarray(resonator_population), # Expectation value of the resonator population (number operator) as a function of time, converted to a NumPy array.
        fluxonium_populations=np.asarray(fluxonium_populations), # Expectation values of the fluxonium populations (number operators) as a function of time, converted to a NumPy array.
        leakage=np.asarray(leakage), # Expectation value of the leakage probability as a function of time, converted to a NumPy array.
    )


def run_readout_pair(
    params: DeviceParameters,
    pulse: PulseParameters,
    sim: SimulationParameters,
) -> ReadoutPairResult:
    # Run readout simulations for both the ground and excited states of the fluxonium, and compute the resulting SNR and assignment error as functions of time. We use the run_readout_trajectory function to simulate the trajectories for both initial states, and then compute the SNR and assignment error based on the resulting resonator fields.
    ground = run_readout_trajectory(params, pulse, sim, initial_fluxonium_state=0) # Simulate the readout trajectory for the ground state of the fluxonium, which corresponds to an initial fluxonium state of 0. We obtain the ReadoutResult for the ground state, which includes the time points, resonator field, resonator population, fluxonium populations, and leakage as functions of time.
    excited = run_readout_trajectory(params, pulse, sim, initial_fluxonium_state=1) # Simulate the readout trajectory for the excited state of the fluxonium, which corresponds to an initial fluxonium state of 1. We obtain the ReadoutResult for the excited state, which includes the time points, resonator field, resonator population, fluxonium populations, and leakage as functions of time.
    snr = readout_snr( # Compute the SNR for the readout, based on the resonator fields for the ground and excited states, and the measurement parameters. The SNR is computed as a function of time, and takes into account the separation between the resonator fields as well as the measurement efficiency and resonator decay rate.
        sim.tsave,
        ground.resonator_field,
        excited.resonator_field,
        params.kappa,
        sim.measurement_efficiency,
    )
    assignment_error = assignment_error_from_snr(snr, sim.tsave, sim.t1) # Compute the assignment error for the readout, based on the SNR and the measurement parameters. The assignment error is computed as a function of time, and includes contributions from both the Gaussian error due to the overlap of the resonator field distributions and the T1 decay if a T1 time is provided.
    return ReadoutPairResult(ground, excited, snr, assignment_error)


def readout_snr(
    tsave: np.ndarray,
    ground_field: np.ndarray,
    excited_field: np.ndarray,
    kappa: float,
    measurement_efficiency: float,
) -> np.ndarray:
    # Compute the signal-to-noise ratio (SNR) for the readout, based on the difference between the resonator fields for the ground and excited states, and the measurement parameters. The SNR is computed as the square root of 2 times kappa times the measurement efficiency times the cumulative integral of the separation between the two fields over time.
    separation = np.abs(excited_field - ground_field) ** 2. # Compute the separation between the resonator fields for the ground and excited states, which is given by the squared magnitude of the difference between the two fields. This represents the signal power that distinguishes the two states in the readout.
    dt = np.diff(tsave) # Compute the time step between the saved time points, which is used for integrating the separation over time to compute the cumulative signal power.
    cumulative = np.zeros_like(tsave, dtype=float) # Initialize an array to hold the cumulative integral of the separation over time, which will be used to compute the SNR. We start with an array of zeros, and we will fill in the values by integrating the separation using the trapezoidal rule.
    if len(tsave) > 1:
        trapezoids = 0.5 * (separation[1:] + separation[:-1]) * dt # Compute the trapezoidal integral of the separation over time, which gives us the cumulative signal power as a function of time. We use the trapezoidal rule to approximate the integral, which is more accurate than a simple rectangular approximation.
        cumulative[1:] = np.cumsum(trapezoids) # Compute the cumulative integral by taking the cumulative sum of the trapezoidal contributions. This gives us the total signal power accumulated up to each time point in tsave.
    return np.sqrt(2.0 * kappa * measurement_efficiency * cumulative) # Compute the SNR as the square root of 2 times kappa times the measurement efficiency times the cumulative integral of the separation. This formula arises from the theory of continuous quantum measurement, where the SNR is proportional to the square root of the total signal power accumulated over time, and is also proportional to the measurement efficiency and the resonator decay rate.


def assignment_error_from_snr(
    snr: np.ndarray,
    tsave: np.ndarray,
    t1: float | None,
) -> np.ndarray:
    # Compute the assignment error for the readout, based on the SNR and the measurement parameters. The assignment error is given by the Gaussian error function of the SNR, plus an additional contribution from T1 decay if t1 is provided. The T1 contribution is computed as tsave divided by 2 times t1, which accounts for the probability of decay during the measurement time.
    gaussian_error = 0.5 * erfc(snr / (2.0 * np.sqrt(2.0))) # Compute the Gaussian error contribution to the assignment error, which is given by the complementary error function of the SNR divided by 2. This represents the probability of misassigning the state based on the overlap of the resonator field distributions for the ground and excited states.
    if t1 is None:
        return gaussian_error # If no T1 time is provided, the assignment error is just given by the Gaussian error contribution.
    return gaussian_error + tsave / (2.0 * t1)


def to_numpy(x) -> np.ndarray:
    # Convert a JAX array to a NumPy array. This is useful for ensuring that the results of the simulations are returned as standard NumPy arrays, which can be more convenient for further analysis and plotting.
    return np.asarray(jnp.asarray(x))