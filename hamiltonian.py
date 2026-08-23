from __future__ import annotations

import jax.numpy as jnp
# import jax
# jax.config.update("jax_enable_x64", True)
import numpy as np
import dynamiqs as dq

from parameters import DeviceParameters, PulseParameters

#from simulation import to_numpy

def to_numpy(x) -> np.ndarray:
    '''A FUNTION TO CONVERT THE JAX array to a NumpPy array'''
    # Convert a JAX array to a NumPy array. This is useful for ensuring that the results of the simulations are returned as standard NumPy arrays, which can be more convenient for further analysis and plotting.
    return np.asarray(jnp.asarray(x))


def operators(params: DeviceParameters, pulse: PulseParameters ) -> dict[str, dq.QArray]:
    '''RETURNS ALL THE OPERATORS REQUIRED 
    identity_f : IDENTITY OPERATOR FOR CONSTRUCTING THE FLUXONIUM OPERATORS
    identity_r : IDENTITY OPERATOR FOR CONSTRUCTING THE RESONATOR OPERATORS
    a_r : DESTROY OPERATOR FOR THE RESONATOR 
    n_f : CHARGE OPEARTOR FOR THE FLUXONIUM SUBSYSTEM
    h_f : HAMILTONIAN FOR THE FLUXONIUM SUBSYSTEM AS A Qarray
    a : ANNIHILATION OPERATOR FOR THE FULL SYSTEM.
    n : CHARGE OPRATOR FOR THE FULL SYSTEM.
    n_plus : CHARGE OPRATOR CONVERTED TO RAISING OPRATOR n+
    n_minus : CHARGE OPRATOR CONVERTED TO LOWERING OPRATOR n+
    h_fluxonium
    '''
    identity_f = dq.eye(params.fluxonium_dim, layout=dq.dia) # Identity operator for the fluxonium subsystem
    identity_r = dq.eye(params.resonator_dim, layout=dq.dia) # Identity operator for the resonator subsystem
    a_r = dq.destroy(params.resonator_dim, layout=dq.dia) # Annihilation operator for the resonator subsystem

    #omega_rot = (params.omega_levels- np.arange(len(params.omega_levels))* pulse.omega_d)

    n_f = dq.asqarray(params.n_matrix, dims=(params.fluxonium_dim,) ) # Charge operator for the fluxonium subsystem, represented as a QArray with appropriate dimensions
    h_f = dq.asqarray(np.diag(params.omega_levels), dims=(params.fluxonium_dim,), layout=dq.dia) # Hamiltonian for the fluxonium subsystem, represented as a QArray with appropriate dimensions


    a = dq.tensor(identity_f, a_r) # Annihilation operator for the full system, constructed as a tensor product of the identity on the fluxonium subsystem and the annihilation operator on the resonator subsystem
    n = dq.tensor(n_f, identity_r) # Charge operator for the full system, constructed as a tensor product of the charge operator on the fluxonium subsystem and the identity on the resonator subsystem
 
    
    h_fluxonium = dq.tensor(h_f, identity_r) # Hamiltonian for the full system, constructed as a tensor product of the Hamiltonian on the fluxonium subsystem and the identity on the resonator subsystem
    h_fluxonium_f = h_f
    h_resonator = params.omega_r * (a.dag() @ a) # Hamiltonian for the resonator subsystem, represented as a QArray
    drive_op = -1j * (a - a.dag()) # Drive operator for the resonator, represented as a QArray. This is the operator that couples to the drive in the Hamiltonian, and is constructed as -i times the difference between the annihilation and creation operators of the resonator.
    #drive_op = a + a.dag() # Drive operator for the resonator, represented as a QArray. This is the operator that couples to the drive in the Hamiltonian, and is constructed as the sum of the annihilation and creation operators of the resonator.
    return {
        "a": a,
        "n": n,
        "n_f": n_f,
        "h_f": h_f,
        "h_fluxonium": h_fluxonium,
        "h_static": h_resonator + h_fluxonium + params.g * drive_op @ n,
    #    "h_static": h_resonator, #+ h_fluxonium + params.g * n @ drive_op,
        "drive_op": drive_op,
        "n_photon": a.dag() @ a,
    }

# def Omega_r(params: DeviceParameters, pulse: PulseParameters, fluxonium_state: int) -> float:
#     ops = operators(params, pulse)
#     h_dressed = ops["h_static"]

#     H = np.asarray(h_dressed.to_jax())
#     evals, evecs = np.linalg.eigh(H)

#     def overlap_idx(i: int, n: int) -> tuple[int, float]:
#         psi = np.asarray(basis_state(params, i, n).to_jax()).flatten()
#         overlaps = np.abs(evecs.conj().T @ psi) ** 2
#         idx = np.argmax(overlaps)
#         return idx, overlaps[idx]

#     idx_g0, ov_g0 = overlap_idx(0, 0)
#     idx_g1, ov_g1 = overlap_idx(0, 1)
#     idx_e0, ov_e0 = overlap_idx(1, 0)
#     idx_e1, ov_e1 = overlap_idx(1, 1)

#     for label, ov in [("g0", ov_g0), ("g1", ov_g1), ("e0", ov_e0), ("e1", ov_e1)]:
#         if ov < 0.9:
#             print(f"WARNING: weak overlap for {label}: {ov:.3f}")

#     omega_r_g = evals[idx_g1] - evals[idx_g0]
#     omega_r_e = evals[idx_e1] - evals[idx_e0]

#    return omega_r_g if fluxonium_state == 0 else omega_r_e

def Omega_r(params: DeviceParameters, pulse: PulseParameters, fluxonium_state: int) -> float:
    '''RETURNS THE CHANGED RESONATOR FREQUENCY DUE TO COUPLING TO THE QUBIT'''
    ops = operators(params, pulse)
    H = np.asarray(ops["h_static"].to_jax())
    evals, evecs = np.linalg.eigh(H)

    a_dag_op = np.asarray(ops["a"].to_jax()).conj().T

    def anchor(i_f: int) -> tuple[int, float]:
        psi = np.asarray(basis_state(params, i_f, 0).to_jax()).flatten()
        overlaps = np.abs(evecs.conj().T @ psi) ** 2
        idx = np.argmax(overlaps)
        return idx, overlaps[idx]

    def next_branch_state(idx_current: int) -> tuple[int, float]:
        target = a_dag_op @ evecs[:, idx_current]
        target /= np.linalg.norm(target)
        overlaps = np.abs(evecs.conj().T @ target) ** 2
        idx = np.argmax(overlaps)
        return idx, overlaps[idx]

    idx_g0, ov_g0 = anchor(0)
    idx_g1, ov_g1 = next_branch_state(idx_g0)
    idx_e0, ov_e0 = anchor(1)
    idx_e1, ov_e1 = next_branch_state(idx_e0)

    for label, ov in [("g0", ov_g0), ("g1", ov_g1), ("e0", ov_e0), ("e1", ov_e1)]:
        if ov < 0.9:
            print(f"WARNING: weak overlap for {label}: {ov:.3f}")

    omega_r_g = evals[idx_g1] - evals[idx_g0]
    omega_r_e = evals[idx_e1] - evals[idx_e0]

    return omega_r_g if fluxonium_state == 0 else omega_r_e
# def readout_hamiltonian(
#     params: DeviceParameters,
#     pulse: PulseParameters,
#     fluxonium_state: int
#     #total_time: float,
# ) -> dq.TimeQArray:
#     ops = operators(params, pulse)
#     omega_r = Omega_r(params, pulse, fluxonium_state)
#     h_static = dq.constant(ops["h_static"])

#     # def modulation(t: float):
#     #     # Compute the time-dependent modulation for the drive term in the Hamiltonian, based on the pulse parameters and the current time t. The modulation is given by the envelope of the pulse multiplied by a cosine function at the drive frequency.
#     #     active = jnp.where((t >= 0.0) & (t <= total_time), 1.0, 0.0) # This ensures that the modulation is only active during the time interval of the pulse, and is zero outside of that interval.
#     #     rise = 1.0 if pulse.rise_time <= 0 else jnp.minimum(1.0, t / pulse.rise_time) # Compute the rise envelope of the pulse, which ramps up from 0 to 1 over the specified rise time. If the rise time is zero or negative, we assume an instantaneous rise and set the envelope to 1.
#     #     fall_start = total_time - pulse.fall_time # Compute the start time of the fall envelope, which is the total pulse duration minus the specified fall time.
#     #     fall = (
#     #         1.0
#     #         if pulse.fall_time <= 0
#     #         else jnp.where(t > fall_start, jnp.maximum(0.0, (total_time - t) / pulse.fall_time), 1.0)
#     #     )
#     #     envelope = pulse.amplitude * jnp.minimum(rise, fall) * active
#     #     return envelope * jnp.cos(pulse.omega_d * t)
#     # def modulation(t: float):
#     #     amp = jnp.where(t< 100.0, pulse.epsilon1, jnp.where(t < 300.0, pulse.epsilon2, 0.0))
#     #     return amp * jnp.cos(pulse.omega_d * t)
#     def modulation(t):

#         t_ramp = 20

#         ramp = pulse.epsilon1 * 0.5 * (
#             1 - jnp.cos(jnp.pi * t / t_ramp)
#         )

#         amp1 = jnp.where(
#             t < t_ramp,
#             ramp,
#             pulse.epsilon1
#         )

#         #amp1 = jnp.where(t<t_ramp, pulse.epsilon1 * t/ t_ramp, pulse.epsilon1)

#         amp = jnp.where(
#             t < 100.0,
#             amp1,
#             jnp.where(t < 300.0,
#                     pulse.epsilon2,
#                     0.0)
#         )

#         return 0.5 * amp

#     return h_static + dq.modulated(modulation, ops["drive_op"])

def readout_hamiltonian(params: DeviceParameters, pulse: PulseParameters, fluxonium_state: int) -> dq.TimeQArray:
    '''READOUT HAMILTONIAN BASED ON THE SHILLITO et.al . PAPER'''
    ops = operators(params, pulse)
    omega_r = Omega_r(params, pulse, fluxonium_state)
    print("FOR FLUXONIUM STATE ", fluxonium_state, "omega_r is ", omega_r )
    #print(omega_r)
    # omega_d = 0.5 * (Omega_r(params, pulse, 1) + Omega_r(params, pulse, 0))
    # omega_d = pulse.omega_d
    omega_d = Omega_r(params, pulse, 1)
    # omega_d = Omega_r(params, pulse, 0)
    omega_r_bare = params.omega_r
    delta_r = omega_r_bare - omega_d
    h_res = delta_r * ops["n_photon"]

    def coup_a(t: float):
        return  params.g * jnp.exp((-1j) * omega_d * t)

    def coup_a_dag(t: float):
        return  params.g * jnp.exp((1j) * omega_d * t)

    h_coupling = (-1j) * (dq.modulated(coup_a_dag, ops["n"] @ ops["a"].dag()) - dq.modulated(coup_a, ops["n"] @ ops["a"]))

    h_static = h_res + ops["h_fluxonium"] + h_coupling

    def modulation(t: float): #based on the pulse parameters and the current time t. The modulation is given by the envelope of the pulse multiplied by a cosine function at the drive frequency.
        amp = jnp.where(t< 100.0, pulse.epsilon1, jnp.where(t < 300.0, pulse.epsilon2, 0.0))
       
        return amp * 0.5

    # def modulation(t: float):
    #     rise_time = 16.0
    #     fall_time = 16.0

    #     amp = jnp.where(
    #         t < 100.0,
    #         pulse.epsilon1,
    #         jnp.where(t < 300.0, pulse.epsilon2, 0.0)
    #     )

    #     rise = jnp.where(
    #         t < rise_time,
    #         jnp.exp(-((t - rise_time)**2)/(2*(rise_time/3)**2)),
    #         1.0,
    #     )

    #     fall = jnp.where(
    #         t > 300.0 - fall_time,
    #         jnp.exp(-((t - (300.0 - fall_time))**2)/(2*(fall_time/3)**2)),
    #         1.0,
    #     )

    #     return 0.5 * amp * rise * fall

    def drive_static(t: float):
        return -modulation(t)                          # coeff of (a + a.dag())

    def drive_a(t: float):
        return modulation(t) * jnp.exp(2 * (-1j) * omega_d * t)   # coeff of a, fast

    def drive_a_dag(t: float):
        return modulation(t) * jnp.exp(2 * (1j) * omega_d * t)    # coeff of a.dag(), fast

    h_drive = (1j)*(dq.modulated(drive_static, ops["a"] - ops["a"].dag())
               + dq.modulated(drive_a, ops["a"])
               - dq.modulated(drive_a_dag, ops["a"].dag()))
    
    

    return h_static + h_drive, omega_d


def floquet_readout_hamiltonian(params: DeviceParameters, pulse: PulseParameters, fluxonium_state: int) -> dq.TimeQArray:
    '''READOUT HAMILTONIAN BASED ON THE SHILLITO et.al . PAPER'''
    ops = operators(params, pulse)
    omega_r = Omega_r(params, pulse, fluxonium_state)
    #print(omega_r)
    omega_d = 0.5 * (Omega_r(params, pulse, 1) + Omega_r(params, pulse, 0))
    #omega_d = Omega_r(params, pulse, fluxonium_state)
    omega_r_bare = params.omega_r
    delta_r = omega_r_bare - omega_d
    h_res = delta_r * ops["n_photon"]

    def wrap_angle(theta):
        return jnp.mod(theta, 2 * jnp.pi)

    def coup_a(t: float):
        p = wrap_angle(omega_d * t)
        return  params.g * jnp.exp((-1j) * round(p, 3))

    def coup_a_dag(t: float):
        p = wrap_angle(omega_d * t)
        return  params.g * jnp.exp((1j) * round(p, 3))

    h_coupling = (-1j) * (dq.modulated(coup_a_dag, ops["n"] @ ops["a"].dag()) - dq.modulated(coup_a, ops["n"] @ ops["a"]))

    h_static = h_res + ops["h_fluxonium"] + h_coupling

    def modulation(t: float): #based on the pulse parameters and the current time t. The modulation is given by the envelope of the pulse multiplied by a cosine function at the drive frequency.
        amp = jnp.where(t< 100.0, pulse.epsilon1, jnp.where(t < 300.0, pulse.epsilon2, 0.0))
       
        return amp * 0.5

    # def modulation(t: float):
    #     rise_time = 16.0
    #     fall_time = 16.0

    #     amp = jnp.where(
    #         t < 100.0,
    #         pulse.epsilon1,
    #         jnp.where(t < 300.0, pulse.epsilon2, 0.0)
    #     )

    #     rise = jnp.where(
    #         t < rise_time,
    #         jnp.exp(-((t - rise_time)**2)/(2*(rise_time/3)**2)),
    #         1.0,
    #     )

    #     fall = jnp.where(
    #         t > 300.0 - fall_time,
    #         jnp.exp(-((t - (300.0 - fall_time))**2)/(2*(fall_time/3)**2)),
    #         1.0,
    #     )

    #     return 0.5 * amp * rise * fall

    def drive_static(t: float):
        return -modulation(t)                          # coeff of (a + a.dag())

    def drive_a(t: float):
        p = wrap_angle(omega_d * t)
        return modulation(t) * jnp.exp(2 * (-1j) * round(p, 3))   # coeff of a, fast

    def drive_a_dag(t: float):
        p = wrap_angle(omega_d * t)
        return modulation(t) * jnp.exp(2 * (1j) * round(p, 3))    # coeff of a.dag(), fast

    h_drive = (1j)*(dq.modulated(drive_static, ops["a"] - ops["a"].dag())
               + dq.modulated(drive_a, ops["a"])
               - dq.modulated(drive_a_dag, ops["a"].dag()))
    
    

    return h_static + h_drive, omega_d

def readout_hamiltonian_rwa(params: DeviceParameters, pulse: PulseParameters, fluxonium_state: int) -> dq.TimeQArray:
    '''READOUT HAMILTONIAN IN THE RWA'''
    ops = operators(params, pulse)
    omega_r = Omega_r(params, pulse, fluxonium_state)
    print(omega_r)
    omega_d = 0.5 * (Omega_r(params, pulse, 1) + Omega_r(params, pulse, 0))
    omega_r_bare = params.omega_r
    delta_r = omega_r_bare - omega_d
    delta_r = omega_r - omega_d #pulse.omega_d # Compute the detuning of the resonator frequency from the drive frequency, which is an important parameter in the rotating frame Hamiltonian. The detuning determines how the resonator responds to the drive, and can lead to different dynamics depending on whether it is positive, negative, or zero.
    h_res = delta_r * ops["n_photon"] # The resonator Hamiltonian in the rotating frame is given by the detuning times the number operator for the resonator. This captures the energy of the resonator photons relative to the drive frequency, and is a key part of the dynamics in the rotating frame.
    h_static = h_res + ops["h_fluxonium"] + params.g *(-1j)*( ops["a"] @ ops["n_plus"] - ops["a"].dag() @ ops["n_minus"] ) # The static part of the Hamiltonian in the rotating frame includes the resonator Hamiltonian, the fluxonium Hamiltonian, and the coupling term between the drive and the charge operator of the fluxonium. This static Hamiltonian captures the essential physics of the system in the rotating frame, and serves as the baseline for adding the time-dependent drive modulation.
    print("Hermicity error:", dq.norm(h_static - h_static.dag()))
    h_static = dq.constant(h_static) # We then add the time-dependent modulation for the drive term, which captures the time dependence of the drive amplitude. The modulation function is defined to have different values during different time intervals of the pulse, allowing us to model a pulse that has a certain amplitude for a specified duration and then turns off. The modulation is applied to the drive operator in the Hamiltonian, which leads to time-dependent dynamics when we simulate the system.
    # def modulation(t: float): #based on the pulse parameters and the current time t. The modulation is given by the envelope of the pulse multiplied by a cosine function at the drive frequency.
    #     amp = jnp.where(t< 100.0, pulse.epsilon1, jnp.where(t < 300.0, pulse.epsilon2, 0.0))
    #    # print(t)
    #     return amp * 0.5
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
    def modulation(t: float):
        rise_time = 96.0
        fall_time = 96.0

        amp = jnp.where(
            t < 100.0,
            pulse.epsilon1,
            jnp.where(t < 300.0, pulse.epsilon2, 0.0)
        )

        rise = jnp.where(
            t < rise_time,
            jnp.exp(-((t - rise_time)**2)/(2*(rise_time/3)**2)),
            1.0,
        )

        fall = jnp.where(
            t > 300.0 - fall_time,
            jnp.exp(-((t - (300.0 - fall_time))**2)/(2*(fall_time/3)**2)),
            1.0,
        )

        return 0.5 * amp * rise * fall
    return h_static + dq.modulated(modulation, ops["drive_op"]), omega_d


def jump_operators(params: DeviceParameters, a: dq.QArray) -> list[dq.QArray]:
    '''JUMP OPERATORS FOR THE mesolve WITH SQUARE ROOT of Resonator Decay and destruction operator'''
    # For now, we only include the resonator decay as a jump operator. If kappa is zero, we return an empty list to indicate no dissipation.
    if params.kappa == 0:
        return []
    return [np.sqrt(params.kappa) * a]


def basis_state(params: DeviceParameters, fluxonium_state: int, resonator_state: int = 0) -> dq.QArray:
    '''CREATES THE INITIAL BASIS STATE'''
    # Construct a basis state for the combined fluxonium-resonator system, given the specified fluxonium and resonator states. The basis state is represented as a QArray, constructed using the tensor product of the fluxonium and resonator basis states.
    return dq.fock((params.fluxonium_dim, params.resonator_dim), [fluxonium_state, resonator_state])


def fluxonium_projector(params: DeviceParameters, fluxonium_state: int) -> dq.QArray:
    '''CREATES A PROJECTION OPERATOR'''
    # Construct a projector onto a specific fluxonium state, represented as a QArray. The projector is constructed as the tensor product of the projector onto the specified fluxonium state and the identity operator on the resonator subsystem.
    ket = dq.fock(params.fluxonium_dim, fluxonium_state)
    bra = dq.fock(params.fluxonium_dim, fluxonium_state).dag()
    projector = ket @ bra
    return dq.tensor(projector, dq.eye(params.resonator_dim, layout=dq.dense))


def computational_projector(params: DeviceParameters) -> dq.QArray:
    '''CREATES A COMPUTATIONAL PROJECTOR ONTO THE BARE QUBIT HILBERT SPACE.'''
    # Construct a projector onto the computational subspace, represented as a QArray. This is the sum of projectors onto the ground and excited states of the fluxonium.
    return fluxonium_projector(params, 0) + fluxonium_projector(params, 1)

# def dressed_fluxonium_computational_projector(params: DeviceParameters, pulse: PulseParameters):
#     ops = operators(params, pulse)

#     # static, driveless, LAB-FRAME Hamiltonian -- matches paper's E=0 case exactly
#     H = ops["h_static"]   # = omega_r * n_photon + h_fluxonium - i*g*(a - a.dag()) @ n

#     H = np.asarray(H.to_jax())
#     evals, evecs = np.linalg.eigh(H)

#     dressed_indices = set()
#     for n in range(params.resonator_dim):
#         g_n = np.asarray(basis_state(params, 0, n).to_jax()).flatten()
#         overlaps = np.abs(evecs.conj().T @ g_n) ** 2
#         dressed_indices.add(np.argmax(overlaps))
#         best_overlap = overlaps[np.argmax(overlaps)]
#         if best_overlap < 0.5:
#             print(f"WARNING: weak identification for |{0 or 1},{n}> -- best overlap = {best_overlap:.3f}")

#         e_n = np.asarray(basis_state(params, 1, n).to_jax()).flatten()
#         overlaps = np.abs(evecs.conj().T @ e_n) ** 2
#         dressed_indices.add(np.argmax(overlaps))
#         if best_overlap < 0.5:
#             print(f"WARNING: weak identification for |{0 or 1},{n}> -- best overlap = {best_overlap:.3f}")

#     P_comp = np.zeros_like(H, dtype=complex)
#     for idx in dressed_indices:
#         psi = evecs[:, idx]
#         P_comp += np.outer(psi, psi.conj())

#     return dq.asqarray(P_comp, dims=(params.fluxonium_dim, params.resonator_dim))

# def dressed_fluxonium_computational_projector(params: DeviceParameters, pulse: PulseParameters):
#     ops = operators(params, pulse)
#     H = np.asarray(ops["h_static"].to_jax())
#     evals, evecs = np.linalg.eigh(H)

#     a_op = np.asarray(ops["a"].to_jax())  # full system a operator, for the a-dagger overlap step
#     a_dag_op = a_op.conj().T

#     n_states = evecs.shape[1]
#     unassigned = set(range(n_states))
#     dressed_indices = set()
#     branch_chain = {0: [], 1: []}  # store the branch for i_f = 0, 1

#     for i_f in (0, 1):
#         # --- anchor step: n_r = 0, direct overlap with bare |i_f, 0_r> ---
#         psi0 = np.asarray(basis_state(params, i_f, 0).to_jax()).flatten()
#         overlaps = np.abs(evecs.conj().T @ psi0) ** 2
#         # restrict search to still-unassigned states to avoid double-claiming
#         overlaps_masked = np.array([overlaps[k] if k in unassigned else -1.0 for k in range(n_states)])
#         idx = np.argmax(overlaps_masked)
#         best_overlap = overlaps_masked[idx]
#         if best_overlap < 0.5:
#             print(f"WARNING: weak anchor identification for |{i_f},0> -- overlap = {best_overlap:.3f}")

#         dressed_indices.add(idx)
#         unassigned.discard(idx)
#         branch_chain[i_f].append(idx)
#         current_psi = evecs[:, idx]

#         # --- recursive step: n_r -> n_r+1 via a-dagger overlap ---
#         for n_r in range(params.resonator_dim - 1):
#             target = a_dag_op @ current_psi   # a^dagger |i_f, n_r-bar>
#             target_norm = np.linalg.norm(target)
#             if target_norm < 1e-12:
#                 print(f"WARNING: a^dagger|psi> ~ 0 at n_r={n_r} for i_f={i_f}, stopping branch")
#                 break
#             target /= target_norm

#             candidate_overlaps = np.array([
#                 np.abs(np.vdot(evecs[:, k], target)) if k in unassigned else -1.0
#                 for k in range(n_states)
#             ])
#             idx_next = np.argmax(candidate_overlaps)
#             best_next = candidate_overlaps[idx_next]
#             if best_next < 0.5:
#                 print(f"WARNING: weak branch-tracking overlap at i_f={i_f}, n_r={n_r+1} -- overlap = {best_next:.3f}")

#             dressed_indices.add(idx_next)
#             unassigned.discard(idx_next)
#             branch_chain[i_f].append(idx_next)
#             current_psi = evecs[:, idx_next]

#     P_comp = np.zeros_like(H, dtype=complex)
#     for idx in dressed_indices:
#         psi = evecs[:, idx]
#         P_comp += np.outer(psi, psi.conj())

    

#     return dq.asqarray(P_comp, dims=(params.fluxonium_dim, params.resonator_dim)), branch_chain

def build_branch_projectors(params: DeviceParameters, pulse: PulseParameters, overlap_warn_thres: float=0.5) -> dict:
    '''DIAGONALIZES THE STATIC LAB-FRAME HAMILTONIAN TO BUILD PER-BRANCH P_B0 AND P_B1 .'''
    
    ops = operators(params, pulse)
    H = np.asarray(ops["h_static"].to_jax())
    
    evals, evecs = np.linalg.eigh(H)
    a_dag_op = np.asarray((ops["a"]).to_jax()).conj().T
    branch_indices = {0: [], 1: []}
    branch_overlaps = {0: [], 1: []}

    ###Here we do the overlap for the first dressed state which is same as the bare state.

    for i_f in (0, 1):
        psi0 = np.asarray((basis_state(params, i_f, 0)).to_jax()).flatten()
        overlap0 = np.abs(evecs.conj().T @ psi0) ** 2 ### checks the overlap , standard method to do it.
        idx = np.argmax(overlap0) ### gives the index of the highest overlap
        if overlap0[idx] < overlap_warn_thres :
            print(f"WARNING WEAK ANCHOR i_f = {i_f}, n_r = 0 : overlap = {overlap0[idx]:.3f}")
        branch_indices[i_f].append(idx)
        branch_overlaps[i_f].append(idx)
        current = evecs[:, idx] ### eigen vector of the current branch with which 0 or 1 has th max overlap

        #####to check overlap with higher states 

        for n_r in range(params.resonator_dim - 1 ):
            target = a_dag_op @ current
            norm = np.linalg.norm(target)
            if norm < 1e-10:
                break
            target /= norm
            cand = np.abs(evecs.conj().T @ target) ** 2
            idx = np.argmax(cand)
            if cand[idx] < overlap_warn_thres:
                print(f"WARNING WEAK TRACKING i_f = {i_f}, n_r = {n_r+1} : overlap = {cand[idx]:.3f}")
            branch_indices[i_f].append(idx)
            branch_overlaps[i_f].append(idx)
            current = evecs[:, idx]

    dim = H.shape[0]
    P_B0 = np.zeros((dim, dim), dtype = complex)
    for idx in branch_indices[0]:
        psi = evecs[:, idx]
        P_B0 += np.outer(psi, psi.conj()) ### population for branch 0

    P_B1 = np.zeros((dim, dim), dtype = complex)
    for idx in branch_indices[1]:
        psi = evecs[:, idx]
        P_B1 += np.outer(psi, psi.conj()) ### population for branch 1

    return {
        "P_B0": P_B0,
        "P_B1": P_B1,
        "branch_indices": branch_indices,
        "branch_overlaps": branch_overlaps,
    }

def dressed_leakage_from_states(states, tvals, P_B0: np.ndarray, P_B1: np.ndarray, omega_d: float, n_op_np: np.ndarray, trace_tol: float = 1e-4 )-> dict:
    '''CALCULATES THE POPULATION OF BRANCHES 0 AND 1 AFTER mesolve'''
    p_B0, p_B1 = [], []
    for t, rho_i in zip(tvals, states):
        rho_I_np = np.asarray(rho_i.to_jax())
        phase = np.exp((1j)* omega_d * n_op_np * t)
        U_rf = np.diag(phase)
        rho_lab = U_rf.conj().T @ rho_I_np @ U_rf
        tr = np.real(np.trace(rho_lab))
        if abs(tr - 1.0)> trace_tol:
            print(f"warning: trach (rho_lab) = { tr:.3f} at t = {t} frame transformation inconsistent.")
        
        p_B0.append(np.real(np.trace(rho_lab @ P_B0)))
        p_B1.append(np.real(np.trace(rho_lab @ P_B1)))

    p_B0 = np.array(p_B0) 
    p_B1 = np.array(p_B1)  
    leakage = np.clip(1 - p_B0 - p_B1, 0.0, 1.0)
    print(leakage)

    return { "p_B0": p_B0,"p_B1": p_B1, "leakage": leakage }


