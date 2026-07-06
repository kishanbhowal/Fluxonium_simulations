from __future__ import annotations

import jax.numpy as jnp
# import jax
# jax.config.update("jax_enable_x64", True)
import numpy as np
import dynamiqs as dq

from parameters import DeviceParameters, PulseParameters


def operators(params: DeviceParameters, pulse: PulseParameters ) -> dict[str, dq.QArray]:
    identity_f = dq.eye(params.fluxonium_dim, layout=dq.dia) # Identity operator for the fluxonium subsystem
    identity_r = dq.eye(params.resonator_dim, layout=dq.dia) # Identity operator for the resonator subsystem
    a_r = dq.destroy(params.resonator_dim, layout=dq.dia) # Annihilation operator for the resonator subsystem

    #omega_rot = (params.omega_levels- np.arange(len(params.omega_levels))* pulse.omega_d)

    n_f = dq.asqarray(params.n_matrix, dims=(params.fluxonium_dim,) ) # Charge operator for the fluxonium subsystem, represented as a QArray with appropriate dimensions
    h_f = dq.asqarray(np.diag(params.omega_levels), dims=(params.fluxonium_dim,), layout=dq.dia) # Hamiltonian for the fluxonium subsystem, represented as a QArray with appropriate dimensions
    n_f_minus = dq.asqarray((params.n_matrix_minus), dims=(params.fluxonium_dim,))
    n_f_plus = dq.asqarray((params.n_matrix_plus), dims=(params.fluxonium_dim,))

    a = dq.tensor(identity_f, a_r) # Annihilation operator for the full system, constructed as a tensor product of the identity on the fluxonium subsystem and the annihilation operator on the resonator subsystem
    n = dq.tensor(n_f, identity_r) # Charge operator for the full system, constructed as a tensor product of the charge operator on the fluxonium subsystem and the identity on the resonator subsystem
    n_plus = dq.tensor(n_f_plus, identity_r)
    n_minus = dq.tensor(n_f_minus, identity_r)
    
    h_fluxonium = dq.tensor(h_f, identity_r) # Hamiltonian for the full system, constructed as a tensor product of the Hamiltonian on the fluxonium subsystem and the identity on the resonator subsystem
    h_resonator = params.omega_r * (a.dag() @ a) # Hamiltonian for the resonator subsystem, represented as a QArray
    drive_op = -1j * (a - a.dag()) # Drive operator for the resonator, represented as a QArray. This is the operator that couples to the drive in the Hamiltonian, and is constructed as -i times the difference between the annihilation and creation operators of the resonator.
    #drive_op = a + a.dag() # Drive operator for the resonator, represented as a QArray. This is the operator that couples to the drive in the Hamiltonian, and is constructed as the sum of the annihilation and creation operators of the resonator.
    return {
        "a": a,
        "n": n,
        "n_plus":n_plus,
        "n_minus":n_minus,

        "h_fluxonium": h_fluxonium,
        "h_static": h_resonator + h_fluxonium + params.g * drive_op @ n,
    #    "h_static": h_resonator, #+ h_fluxonium + params.g * n @ drive_op,
        "drive_op": drive_op,
        "n_photon": a.dag() @ a,
    }

def Omega_r(params: DeviceParameters, pulse : PulseParameters, fluxonium_state : int) -> int:
    ops = operators(params, pulse)
    h_static = ops["h_static"]
    h_interaction_rwa = params.g * (-1j)*( ops["n_plus"] @ ops["a"] - ops["n_minus"] @ (ops["a"].dag()))
    h_dressed_rwa = h_static + h_interaction_rwa
    h_interaction = params.g * (-1j)*(ops["a"] - ops["a"].dag()) @ ops["n"]
    h_dressed = h_static + h_interaction
    diagonalized = np.diag(h_dressed)
    H = h_dressed.to_jax()
    H = np.asarray(H)    
    evals, evecs = np.linalg.eigh(H)
    g0 = basis_state(params , 0, 0)
    g1 = basis_state(params , 0, 1)
    e0 = basis_state(params , 1, 0)
    e1 = basis_state(params , 1, 1)
    overlaps = np.abs(evecs.conj().T @ g0)**2
    idx_g0 = np.argmax(overlaps)
    overlaps = np.abs(evecs.conj().T @ g1)**2
    idx_g1 = np.argmax(overlaps)
    overlaps = np.abs(evecs.conj().T @ e0)**2
    idx_e0 = np.argmax(overlaps)
    overlaps = np.abs(evecs.conj().T @ e1)**2
    idx_e1 = np.argmax(overlaps)
    omega_r_g = evals[idx_g1] - evals[idx_g0]
    omega_r_e = evals[idx_e1] - evals[idx_e0]
    
    if fluxonium_state == 0:
        return omega_r_g
    elif fluxonium_state == 1 :
        return omega_r_e
            
def readout_hamiltonian(
    params: DeviceParameters,
    pulse: PulseParameters,
    fluxonium_state: int
    #total_time: float,
) -> dq.TimeQArray:
    ops = operators(params, pulse)
    omega_r = Omega_r(params, pulse, fluxonium_state)
    h_static = dq.constant(ops["h_static"])

    # def modulation(t: float):
    #     # Compute the time-dependent modulation for the drive term in the Hamiltonian, based on the pulse parameters and the current time t. The modulation is given by the envelope of the pulse multiplied by a cosine function at the drive frequency.
    #     active = jnp.where((t >= 0.0) & (t <= total_time), 1.0, 0.0) # This ensures that the modulation is only active during the time interval of the pulse, and is zero outside of that interval.
    #     rise = 1.0 if pulse.rise_time <= 0 else jnp.minimum(1.0, t / pulse.rise_time) # Compute the rise envelope of the pulse, which ramps up from 0 to 1 over the specified rise time. If the rise time is zero or negative, we assume an instantaneous rise and set the envelope to 1.
    #     fall_start = total_time - pulse.fall_time # Compute the start time of the fall envelope, which is the total pulse duration minus the specified fall time.
    #     fall = (
    #         1.0
    #         if pulse.fall_time <= 0
    #         else jnp.where(t > fall_start, jnp.maximum(0.0, (total_time - t) / pulse.fall_time), 1.0)
    #     )
    #     envelope = pulse.amplitude * jnp.minimum(rise, fall) * active
    #     return envelope * jnp.cos(pulse.omega_d * t)
    # def modulation(t: float):
    #     amp = jnp.where(t< 100.0, pulse.epsilon1, jnp.where(t < 300.0, pulse.epsilon2, 0.0))
    #     return amp * jnp.cos(pulse.omega_d * t)
    def modulation(t):

        t_ramp = 20

        ramp = pulse.epsilon1 * 0.5 * (
            1 - jnp.cos(jnp.pi * t / t_ramp)
        )

        amp1 = jnp.where(
            t < t_ramp,
            ramp,
            pulse.epsilon1
        )

        #amp1 = jnp.where(t<t_ramp, pulse.epsilon1 * t/ t_ramp, pulse.epsilon1)

        amp = jnp.where(
            t < 100.0,
            amp1,
            jnp.where(t < 300.0,
                    pulse.epsilon2,
                    0.0)
        )

        return 0.5 * amp

    return h_static + dq.modulated(modulation, ops["drive_op"])

def readout_hamiltonian_rwa(params: DeviceParameters, pulse: PulseParameters, fluxonium_state: int) -> dq.TimeQArray:
    ops = operators(params, pulse) # We first compute the static part of the Hamiltonian in the rotating frame, which includes the detuning of the resonator and the fluxonium Hamiltonian, as well as the coupling term. The drive term is then added as a time-dependent modulation on top of this static Hamiltonian. The modulation function is defined to capture the time dependence of the drive amplitude, which can have different values during different time intervals of the pulse.
    omega_r = Omega_r(params, pulse, fluxonium_state)
    omega_d = 0.5 * (Omega_r(params, pulse , 1) + Omega_r(params, pulse , 0))
    print(omega_r, omega_d)
    delta_r = omega_r - omega_d #pulse.omega_d # Compute the detuning of the resonator frequency from the drive frequency, which is an important parameter in the rotating frame Hamiltonian. The detuning determines how the resonator responds to the drive, and can lead to different dynamics depending on whether it is positive, negative, or zero.
    h_res = delta_r * ops["n_photon"] # The resonator Hamiltonian in the rotating frame is given by the detuning times the number operator for the resonator. This captures the energy of the resonator photons relative to the drive frequency, and is a key part of the dynamics in the rotating frame.
    
    h_static = h_res + ops["h_fluxonium"] + params.g *(-1j)*( ops["a"] * jnp.exp((1j)*omega_d*t) - ops["a"].dag() * jnp.exp((-1j)*omega_d*t)) @ ops["n"]# The static part of the Hamiltonian in the rotating frame includes the resonator Hamiltonian, the fluxonium Hamiltonian, and the coupling term between the drive and the charge operator of the fluxonium. This static Hamiltonian captures the essential physics of the system in the rotating frame, and serves as the baseline for adding the time-dependent drive modulation.
    print("Hermicity error:", dq.norm(h_static - h_static.dag()))
    h_static = dq.constant(h_static) # We then add the time-dependent modulation for the drive term, which captures the time dependence of the drive amplitude. The modulation function is defined to have different values during different time intervals of the pulse, allowing us to model a pulse that has a certain amplitude for a specified duration and then turns off. The modulation is applied to the drive operator in the Hamiltonian, which leads to time-dependent dynamics when we simulate the system.
    def modulation(t: float): #based on the pulse parameters and the current time t. The modulation is given by the envelope of the pulse multiplied by a cosine function at the drive frequency.
        amp = jnp.where(t< 100.0, pulse.epsilon1, jnp.where(t < 300.0, pulse.epsilon2, 0.0))
        print(t)
        return amp * 0.5
    # def modulation(t):

    #     t_ramp = 10

    #     ramp = pulse.epsilon1 * 0.5 * (
    #         1 - jnp.cos(jnp.pi * t / t_ramp)
    #     )

    #     amp1 = jnp.where(
    #         t < t_ramp,
    #         ramp,
    #         pulse.epsilon1
    #     )

    #     amp1 = jnp.where(t<t_ramp, pulse.epsilon1 * t/ t_ramp, pulse.epsilon1)

    #     amp = jnp.where(
    #         t < 100.0,
    #         amp1,
    #         jnp.where(t < 300.0,
    #                 pulse.epsilon2,
    #                 0.0)
    #     )

    #     return 0.5 * amp
    # def modulation(t):

    #     sigma_rise = 100.0   # ns
    #     sigma_step = 1.0    # ns

    #     rise = 0.5*(1 + jnp.tanh((t - 10.0)/sigma_rise))

    #     s12 = 0.5*(1 + jnp.tanh((t - 10.0)/sigma_step))

    #     amp = pulse.epsilon1*(1 - s12) + pulse.epsilon2*s12

    #     #off = 0.5*(1 - jnp.tanh((t - 300.0)/sigma_step))

    #     return 0.5 * amp * rise 
    # def modulation(t):

    #     tau_rise = 20.0  # ns

    #     # Gaussian rise
    #     rise = jnp.where(
    #         t < tau_rise,
    #         jnp.exp(-0.5*((t - tau_rise)/8.0)**2),
    #         1.0
    #     )

    #     rise = (rise - jnp.exp(-0.5*(tau_rise/8.0)**2)) / (
    #             1.0 - jnp.exp(-0.5*(tau_rise/8.0)**2))

    #     # Smooth epsilon1 -> epsilon2 transition
    #     sigma = 5.0
    #     s = 0.5 * (1 + jnp.tanh((t - 100.0)/sigma))

    #     amp = pulse.epsilon1*(1-s) + pulse.epsilon2*s

    #     # Smooth turn-off at 300 ns
    #     off = 0.5 * (1 - jnp.tanh((t - 300.0)/sigma))

    #     return 0.5 * amp * rise * off
    return h_static + dq.modulated(modulation, ops["drive_op"])

def readout_hamiltonian_rwa(params: DeviceParameters, pulse: PulseParameters, fluxonium_state: int) -> dq.TimeQArray:
    ops = operators(params, pulse) # We first compute the static part of the Hamiltonian in the rotating frame, which includes the detuning of the resonator and the fluxonium Hamiltonian, as well as the coupling term. The drive term is then added as a time-dependent modulation on top of this static Hamiltonian. The modulation function is defined to capture the time dependence of the drive amplitude, which can have different values during different time intervals of the pulse.
    omega_r = Omega_r(params, pulse, fluxonium_state)
    omega_r_bare = params.omega_r
    omega_d = 0.5 * (Omega_r(params, pulse , 1) + Omega_r(params, pulse , 0))
    print(omega_r, omega_d)
    delta_r = omega_r - omega_d #pulse.omega_d # Compute the detuning of the resonator frequency from the drive frequency, which is an important parameter in the rotating frame Hamiltonian. The detuning determines how the resonator responds to the drive, and can lead to different dynamics depending on whether it is positive, negative, or zero.
    h_res = delta_r * ops["n_photon"] # The resonator Hamiltonian in the rotating frame is given by the detuning times the number operator for the resonator. This captures the energy of the resonator photons relative to the drive frequency, and is a key part of the dynamics in the rotating frame.
    
    #h_static = h_res + ops["h_fluxonium"] + params.g *(-1j)*( ops["a"] * jnp.exp((1j)*omega_d*t) - ops["a"].dag() * jnp.exp((-1j)*omega_d*t)) @ ops["n"]# The static part of the Hamiltonian in the rotating frame includes the resonator Hamiltonian, the fluxonium Hamiltonian, and the coupling term between the drive and the charge operator of the fluxonium. This static Hamiltonian captures the essential physics of the system in the rotating frame, and serves as the baseline for adding the time-dependent drive modulation.
    
    h_static = dq.constant(h_static) # We then add the time-dependent modulation for the drive term, which captures the time dependence of the drive amplitude. The modulation function is defined to have different values during different time intervals of the pulse, allowing us to model a pulse that has a certain amplitude for a specified duration and then turns off. The modulation is applied to the drive operator in the Hamiltonian, which leads to time-dependent dynamics when we simulate the system.
    
    
    def coup_a(t: float):
        x = jnp.exp((-1j) * omega_d * t)
        #return (-1j) * params.g * ops["n"] @ ops["a"].dag 
        return (-1j) * params.g * x
    
    def coup_a_dag(t: float):
        x = jnp.exp((1j)*omega_d * t)
        return (-1j) * params.g * x
    
    h_coupling = dq.modulated(coup_a_dag, ops["n"]@ ops["a"].dag()) + dq.modulated(coup_a, ops["n"]@ ops["a"])

    h_static = h_res + ops["h_fluxonium"] + h_coupling
    print("Hermicity error:", dq.norm(h_static - h_static.dag()))
    
    
    def modulation(t: float): #based on the pulse parameters and the current time t. The modulation is given by the envelope of the pulse multiplied by a cosine function at the drive frequency.
        amp = jnp.where(t< 100.0, pulse.epsilon1, jnp.where(t < 300.0, pulse.epsilon2, 0.0))
        print(t)
        return amp * 0.5
    
    def drive_a(t: float):
        x = modulation(t)*jnp.exp(2* (-1j)* omega_d * t)
        return x 
    
    def drive_a_dag(t:float):
        x = modulation(t)*jnp.exp(2 * (1j) * omega_d * t)
        return x 
    
    h_drive = dq.modulated(modulation, ops["drive"]) + dq.modulated(drive_a, ops["a"] ) + dq.modulated(drive_a_dag, ops["a"].dag)

    return h_static + h_drive
    




def jump_operators(params: DeviceParameters, a: dq.QArray) -> list[dq.QArray]:
    # For now, we only include the resonator decay as a jump operator. If kappa is zero, we return an empty list to indicate no dissipation.
    if params.kappa == 0:
        return []
    return [np.sqrt(params.kappa) * a]


def basis_state(params: DeviceParameters, fluxonium_state: int, resonator_state: int = 0) -> dq.QArray:
    # Construct a basis state for the combined fluxonium-resonator system, given the specified fluxonium and resonator states. The basis state is represented as a QArray, constructed using the tensor product of the fluxonium and resonator basis states.
    return dq.fock((params.fluxonium_dim, params.resonator_dim), [fluxonium_state, resonator_state])


def fluxonium_projector(params: DeviceParameters, fluxonium_state: int) -> dq.QArray:
    # Construct a projector onto a specific fluxonium state, represented as a QArray. The projector is constructed as the tensor product of the projector onto the specified fluxonium state and the identity operator on the resonator subsystem.
    ket = dq.fock(params.fluxonium_dim, fluxonium_state)
    bra = dq.fock(params.fluxonium_dim, fluxonium_state).dag()
    projector = ket @ bra
    return dq.tensor(projector, dq.eye(params.resonator_dim, layout=dq.dense))


def computational_projector(params: DeviceParameters) -> dq.QArray:
    # Construct a projector onto the computational subspace, represented as a QArray. This is the sum of projectors onto the ground and excited states of the fluxonium.
    return fluxonium_projector(params, 0) + fluxonium_projector(params, 1)

def dressed_fluxonium_computational_projector(params:DeviceParameters, pulse:PulseParameters, fluxonium_state: int): 

    ops = operators(params, pulse)

    omega_r = Omega_r(params, pulse, fluxonium_state)

    h_resonator = omega_r * ops["n_photon"]
    h_fluxonium = ops["h_fluxonium"]

    h_interaction_rwa = (params.g * (-1j)* (ops["a"] @ ops["n_plus"]- ops["a"].dag() @ ops["n_minus"]))

    H = h_resonator + h_fluxonium + h_interaction_rwa

    H = np.asarray(H.to_jax())

    evals, evecs = np.linalg.eigh(H)

    dressed_indices = set()

    for n in range(params.resonator_dim):

        # bare |0,n>
        g_n = np.asarray(basis_state(params, 0, n).to_jax()).flatten()

        overlaps = np.abs(evecs.conj().T @ g_n)**2
        dressed_indices.add(np.argmax(overlaps))

        # bare |1,n>
        e_n = np.asarray(basis_state(params, 1, n).to_jax()).flatten()

        overlaps = np.abs(evecs.conj().T @ e_n)**2
        dressed_indices.add(np.argmax(overlaps))

    P_comp = np.zeros_like(H, dtype=complex)

    for idx in dressed_indices:

        psi = evecs[:, idx]

        P_comp += np.outer(psi,psi.conj())

    return dq.asqarray(P_comp,dims=(params.fluxonium_dim,params.resonator_dim))