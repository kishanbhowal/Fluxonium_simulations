from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import dynamiqs as dq

from parameters import DeviceParameters, PulseParameters


def operators(params: DeviceParameters) -> dict[str, dq.QArray]:
    identity_f = dq.eye(params.fluxonium_dim, layout=dq.dia) # Identity operator for the fluxonium subsystem
    identity_r = dq.eye(params.resonator_dim, layout=dq.dia) # Identity operator for the resonator subsystem
    a_r = dq.destroy(params.resonator_dim, layout=dq.dia) # Annihilation operator for the resonator subsystem

    n_f = dq.asqarray(params.n_matrix, dims=(params.fluxonium_dim,) ) # Charge operator for the fluxonium subsystem, represented as a QArray with appropriate dimensions
    h_f = dq.asqarray(np.diag(params.omega_levels)*np.pi*2, dims=(params.fluxonium_dim,), layout=dq.dia) # Hamiltonian for the fluxonium subsystem, represented as a QArray with appropriate dimensions
    n_f_minus = dq.asqarray(params.n_minus_matrix, dims=(params.fluxonium_dim,) )
    n_f_plus = dq.asqarray(params.n_plus_matrix, dims=(params.fluxonium_dim,) )

    a = dq.tensor(identity_f, a_r) # Annihilation operator for the full system, constructed as a tensor product of the identity on the fluxonium subsystem and the annihilation operator on the resonator subsystem
    n = dq.tensor(n_f, identity_r) # Charge operator for the full system, constructed as a tensor product of the charge operator on the fluxonium subsystem and the identity on the resonator subsystem
    n_minus = dq.tensor(n_f_minus, identity_r)
    n_plus = dq.tensor(n_f_plus, identity_r)
    h_fluxonium = dq.tensor(h_f, identity_r) # Hamiltonian for the full system, constructed as a tensor product of the Hamiltonian on the fluxonium subsystem and the identity on the resonator subsystem
    h_resonator = params.omega_r * (a.dag() @ a) # Hamiltonian for the resonator subsystem, represented as a QArray
    drive_op = -1j * (a - a.dag()) # Drive operator for the resonator, represented as a QArray. This is the operator that couples to the drive in the Hamiltonian, and is constructed as -i times the difference between the annihilation and creation operators of the resonator.
    #drive_op = a + a.dag() # Drive operator for the resonator, represented as a QArray. This is the operator that couples to the drive in the Hamiltonian, and is constructed as the sum of the annihilation and creation operators of the resonator.
    return {
        "a": a,
        "n": n,
        "n_plus": n_plus,
        "n_minus": n_minus,
        "h_fluxonium": h_fluxonium,
        "h_static": h_resonator + h_fluxonium + params.g * drive_op @ n,
    #    "h_static": h_resonator, #+ h_fluxonium + params.g * n @ drive_op,
        "drive_op": drive_op,
        "n_photon": a.dag() @ a,
    }


def readout_hamiltonian(
    params: DeviceParameters,
    pulse: PulseParameters,
    #total_time: float,
) -> dq.TimeQArray:
    ops = operators(params)
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
    def modulation(t: float):
        amp = jnp.where(t< 100.0, pulse.epsilon1, jnp.where(t < 300.0, pulse.epsilon2, 0.0)) * 0
        return amp * jnp.cos(pulse.omega_d * t)

    return h_static + dq.modulated(modulation, ops["drive_op"])

def h_interaction(params: DeviceParameters, pulse: PulseParameters, t:float = 600.0) -> dq.TimeQArray :
        ops = operators(params)
        delta_r = params.omega_r - pulse.omega_d
        amp = jnp.where(t< 100.0, pulse.epsilon1, jnp.where(t < 300.0, pulse.epsilon2, 0.0))
        def modulation(t: float): #based on the pulse parameters and the current time t. The modulation is given by the envelope of the pulse multiplied by a cosine function at the drive frequency.
            amp = jnp.where(t< 100.0, pulse.epsilon1, jnp.where(t < 300.0, pulse.epsilon2, 0.0))
            return amp  
        h_int = params.g *(-1j)*(ops["a"].dag()* jnp.exp(1j * delta_r * t) - ops["a"]* jnp.exp(-1j * delta_r * t)) #+ 0.5 * modulation * (ops["a"].dag()* jnp.exp(2j * pulse.omega_d * t) - ops["a"]* jnp.exp(-2j * pulse.omega_d * t))
        return {"h_interaction" : h_int} 

def readout_hamiltonian_rwa(params: DeviceParameters, pulse: PulseParameters) -> dq.TimeQArray:
    ops = operators(params) # We first compute the static part of the Hamiltonian in the rotating frame, which includes the detuning of the resonator and the fluxonium Hamiltonian, as well as the coupling term. The drive term is then added as a time-dependent modulation on top of this static Hamiltonian. The modulation function is defined to capture the time dependence of the drive amplitude, which can have different values during different time intervals of the pulse.
    h_int = h_interaction(params, pulse)
    delta_r = params.omega_r - pulse.omega_d # Compute the detuning of the resonator frequency from the drive frequency, which is an important parameter in the rotating frame Hamiltonian. The detuning determines how the resonator responds to the drive, and can lead to different dynamics depending on whether it is positive, negative, or zero.
    h_res = delta_r * ops["n_photon"] # The resonator Hamiltonian in the rotating frame is given by the detuning times the number operator for the resonator. This captures the energy of the resonator photons relative to the drive frequency, and is a key part of the dynamics in the rotating frame.
    h_static = h_res + ops["h_fluxonium"] + params.g*(-1j) * (ops["a"]@ ops["n_plus"] - ops["a"].dag()@ ops["n_minus"])  # The static part of the Hamiltonian in the rotating frame includes the resonator Hamiltonian, the fluxonium Hamiltonian, and the coupling term between the drive and the charge operator of the fluxonium. This static Hamiltonian captures the essential physics of the system in the rotating frame, and serves as the baseline for adding the time-dependent drive modulation.
    h_static = dq.constant(h_static) # We then add the time-dependent modulation for the drive term, which captures the time dependence of the drive amplitude. The modulation function is defined to have different values during different time intervals of the pulse, allowing us to model a pulse that has a certain amplitude for a specified duration and then turns off. The modulation is applied to the drive operator in the Hamiltonian, which leads to time-dependent dynamics when we simulate the system.
    def modulation(t: float): #based on the pulse parameters and the current time t. The modulation is given by the envelope of the pulse multiplied by a cosine function at the drive frequency.
        amp = jnp.where(t< 100.0, pulse.epsilon1, jnp.where(t < 300.0, pulse.epsilon2 , 0.0)) 
        #h_interaction = params.g*(-1j) *(-1j)*(ops["a"].dag()* jnp.exp(1j * pulse.omega_d * t) - ops["a"]* jnp.exp(-1j * pulse.omega_d * t)) + 0.5 * amp * (ops["a"].dag()* jnp.exp(2j * pulse.omega_d * t) - ops["a"]* jnp.exp(-2j * pulse.omega_d * t))

        print(f"t={t}, amp={amp}")
        return amp * 0.5
    
    return h_static  + dq.modulated(modulation, ops["drive_op"]) #+ h_int["h_interaction"]


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
    projector = ket @ ket.dag()
    return dq.tensor(projector, dq.eye(params.resonator_dim, layout=dq.dia))


def computational_projector(params: DeviceParameters) -> dq.QArray:
    # Construct a projector onto the computational subspace, represented as a QArray. This is the sum of projectors onto the ground and excited states of the fluxonium.
    return fluxonium_projector(params, 0) + fluxonium_projector(params, 1)