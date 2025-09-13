# Optimized implementation: vectorized gradient computation for mixture+entangling ansatz.
# Reduced total_time and larger dt to speed up while preserving comparison.
import numpy as np, math, itertools, time
import matplotlib.pyplot as plt, pandas as pd
from pathlib import Path
np.set_printoptions(precision=4, suppress=True)

start_time = time.time()

# --- Problem setup ---
N = 3
m = 3
d = 2**m
x_min, x_max = -4.0, 4.0
dx = (x_max - x_min) / d
hbar_eff = 1.0
masses = [1.0]*N
G = 1.0
epsilon = 0.5

x_grid = np.array([x_min + k * dx for k in range(d)])

# Multi-index helpers and ks_matrix, pos_vectors
D = d**N
def idx_to_multi(idx, base=d, length=N):
    out = []
    for _ in range(length):
        out.append(idx % base)
        idx //= base
    return tuple(out)
ks_matrix = np.zeros((D, N), dtype=int)
pos_vectors = np.zeros((D, N))
for idx in range(D):
    ks = idx_to_multi(idx)
    ks_matrix[idx,:] = ks
    pos_vectors[idx,:] = x_grid[list(ks)]

# QFT and p values; build F_total
omega = np.exp(-2j * np.pi / d)
F = np.array([[omega**(q*k) for k in range(d)] for q in range(d)], dtype=complex) / np.sqrt(d)
p_vals = (2 * np.pi * hbar_eff) / (d * dx) * (np.arange(d) - d/2)
F_total = F
for _ in range(N-1):
    F_total = np.kron(F_total, F)

# Potential and kinetic diagonals
V_diag = np.zeros(D, dtype=float)
for idx in range(D):
    xs = pos_vectors[idx,:]
    Vsum = 0.0
    for i,j in itertools.combinations(range(N),2):
        r = abs(xs[i]-xs[j])
        Vsum += -G * masses[i] * masses[j] / math.sqrt(r**2 + epsilon**2)
    V_diag[idx] = Vsum
T_mom_diag = np.zeros(D, dtype=float)
for idx in range(D):
    qs = ks_matrix[idx,:]
    Tsum = 0.0
    for i in range(N):
        pq = p_vals[qs[i]]
        Tsum += pq**2 / (2.0 * masses[i])
    T_mom_diag[idx] = Tsum

def apply_H(psi):
    Vpart = V_diag * psi
    psi_mom = F_total @ psi
    Tpsi_mom = T_mom_diag * psi_mom
    Tpart = F_total.conj().T @ Tpsi_mom
    return Tpart + Vpart

# --- Mixture + entangling ansatz optimized ---
params_per_body = 7
n_params = params_per_body * N + 3

def single_phi_component(x_vals, mu, p, s):
    return np.exp(-0.5 * ((x_vals - mu) / s)**2 + 1j * p * x_vals)

def assemble_components(theta):
    # returns per_body_a (list length N of arrays length d) and per_body_da list of lists
    body_theta = theta[:params_per_body * N].reshape(N, params_per_body)
    per_body_a = []
    per_body_da = []
    for i in range(N):
        alogit, mu1, p1, logs1, mu2, p2, logs2 = body_theta[i]
        s1 = math.exp(logs1); s2 = math.exp(logs2)
        w1 = 1.0 / (1.0 + math.exp(-alogit)); w2 = 1.0 - w1
        phi1 = single_phi_component(x_grid, mu1, p1, s1)
        phi2 = single_phi_component(x_grid, mu2, p2, s2)
        ai = w1 * phi1 + w2 * phi2
        per_body_a.append(ai)
        dalogit = (w1*(1-w1)) * (phi1 - phi2)
        dmu1 = w1 * (phi1 * ((x_grid - mu1) / (s1**2)))
        dp1 = w1 * (1j * x_grid * phi1)
        dlogs1 = w1 * (phi1 * ((x_grid - mu1)**2 / (s1**3))) * s1
        dmu2 = w2 * (phi2 * ((x_grid - mu2) / (s2**2)))
        dp2 = w2 * (1j * x_grid * phi2)
        dlogs2 = w2 * (phi2 * ((x_grid - mu2)**2 / (s2**3))) * s2
        per_body_da.append([dalogit, dmu1, dp1, dlogs1, dmu2, dp2, dlogs2])
    return per_body_a, per_body_da

def build_phi_total(per_body_a):
    phi_total = per_body_a[0]
    for i in range(1, N):
        phi_total = np.kron(phi_total, per_body_a[i])
    return phi_total

def assemble_joint_phi_entangled_fast(theta):
    per_body_a, per_body_da = assemble_components(theta)
    phi_total = build_phi_total(per_body_a)
    alphas = theta[params_per_body * N : params_per_body * N + 3]
    alpha01, alpha02, alpha12 = alphas
    phase = alpha01 * pos_vectors[:,0] * pos_vectors[:,1] + alpha02 * pos_vectors[:,0] * pos_vectors[:,2] + alpha12 * pos_vectors[:,1] * pos_vectors[:,2]
    ent_factor = np.exp(1j * phase)
    phi_ent = phi_total * ent_factor
    norm = np.linalg.norm(phi_ent)
    psi = phi_ent / norm
    return phi_total, phi_ent, psi, per_body_a, per_body_da, ent_factor, norm

def grads_mixture_entangled_fast(theta):
    phi_total, phi_ent, psi, per_body_a, per_body_da, ent_factor, norm = assemble_joint_phi_entangled_fast(theta)
    # Build Ai_vals and DA_vals arrays of shape (D, ) for each body/param using ks_matrix
    # For each body i, Ai_vals = per_body_a[i][ ks_matrix[:,i] ]
    A_vals = np.zeros((D, N), dtype=complex)
    for i in range(N):
        A_vals[:,i] = per_body_a[i][ ks_matrix[:,i] ]
    # DA for each body param per body: shape (N, 7, D)
    DA_vals = np.zeros((N, params_per_body, D), dtype=complex)
    for i in range(N):
        for j in range(params_per_body):
            DA_vals[i,j,:] = per_body_da[i][j][ ks_matrix[:,i] ]
    # grads_phi_total: for global param index param_idx = i*7 + j, vector = phi_total * (DA_vals[i,j,:] / A_vals[:,i])
    grads_phi_ent = np.zeros((n_params, D), dtype=complex)
    phi_total_vec = phi_total  # length D
    # compute mixture param grads (first params_per_body * N)
    for i in range(N):
        Ai = A_vals[:,i]
        for j in range(params_per_body):
            param_idx = i*params_per_body + j
            numer = DA_vals[i,j,:]
            # avoid numerical div by small Ai: where Ai small, set ratio zero (component negligible)
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = np.where(np.abs(Ai) > 1e-12, numer / Ai, 0.0)
            grads_phi_ent[param_idx,:] = (phi_total_vec * ratio) * ent_factor
    # alpha derivatives
    base_alpha_idx = params_per_body * N
    grads_phi_ent[base_alpha_idx + 0, :] = 1j * (pos_vectors[:,0] * pos_vectors[:,1]) * phi_ent
    grads_phi_ent[base_alpha_idx + 1, :] = 1j * (pos_vectors[:,0] * pos_vectors[:,2]) * phi_ent
    grads_phi_ent[base_alpha_idx + 2, :] = 1j * (pos_vectors[:,1] * pos_vectors[:,2]) * phi_ent
    # convert to normalized grads
    grads = np.zeros_like(grads_phi_ent)
    for k in range(n_params):
        dphi = grads_phi_ent[k,:]
        inner = np.vdot(psi, dphi)
        grads[k,:] = dphi / norm - psi * np.real(inner)
    return grads, psi

# McLachlan RHS
def mc_rhs_from_grads(grads, psi):
    n = grads.shape[0]
    A = np.zeros((n,n), dtype=float)
    for k in range(n):
        for l in range(n):
            A[k,l] = np.real(np.vdot(grads[k], grads[l]))
    Hpsi = apply_H(psi)
    C = np.zeros(n, dtype=float)
    for k in range(n):
        C[k] = np.imag(np.vdot(grads[k], Hpsi))
    A += 1e-9 * np.eye(n)
    dot = np.linalg.solve(A, C)
    return dot

# Trotter baseline initial state (product Gaussians)
def gaussian_amplitudes(center, sigma=0.7):
    amp = np.exp(-0.5 * ((x_grid - center) / sigma)**2)
    amp = amp / np.linalg.norm(amp)
    return amp.astype(complex)
centers = [-2.0, 0.0, 2.0]
psis_reg = [gaussian_amplitudes(c, sigma=0.7) for c in centers]
psi0 = psis_reg[0]
for j in range(1, N):
    psi0 = np.kron(psi0, psis_reg[j])
psi0 = psi0 / np.linalg.norm(psi0)

def apply_potential_phase(state, dt_half):
    return state * np.exp(-1j * V_diag * dt_half)
def apply_kinetic_phase_via_qft(state, dt):
    psi_mom = F_total @ state
    psi_mom = psi_mom * np.exp(-1j * T_mom_diag * dt)
    return F_total.conj().T @ psi_mom
def trotter_step(state, dt):
    state = apply_potential_phase(state, dt/2.0)
    state = apply_kinetic_phase_via_qft(state, dt)
    state = apply_potential_phase(state, dt/2.0)
    return state / np.linalg.norm(state)

# Simulation params: smaller total_time and larger dt to reduce compute time
total_time = 0.5
dt = 2e-3
n_steps = int(total_time / dt)
record_every = 5

# Initial theta: mixture params + zero alphas
theta0 = np.zeros(n_params)
for i in range(N):
    base = i*params_per_body
    theta0[base + 0] = 0.0  # alogit => equal weights
    theta0[base + 1] = centers[i] - 0.2  # mu1
    theta0[base + 2] = 0.0  # p1
    theta0[base + 3] = math.log(0.6)  # log s1
    theta0[base + 4] = centers[i] + 0.2  # mu2
    theta0[base + 5] = 0.0  # p2
    theta0[base + 6] = math.log(0.9)  # log s2
theta0[params_per_body * N : params_per_body * N + 3] = np.array([0.0, 0.0, 0.0])

# Run Trotter baseline (shorter)
psi = psi0.copy()
times_t = [0.0]; exps_t = [np.array([np.sum(np.abs(psi)**2 * pos_vectors[:,i]) for i in range(N)])]
for step in range(1, n_steps+1):
    psi = trotter_step(psi, dt)
    if step % record_every == 0:
        times_t.append(step*dt)
        exps_t.append(np.array([np.sum(np.abs(psi)**2 * pos_vectors[:,i]) for i in range(N)]))
exps_t = np.array(exps_t); times_t = np.array(times_t)

# Run mixture+entangling McLachlan with RK4 (optimized)
theta = theta0.copy()
phi_total0, phi_ent0, psi0_var, _, _, _, _ = assemble_joint_phi_entangled_fast(theta)
times_m = [0.0]; exps_m = [np.array([np.sum(np.abs(psi0_var)**2 * pos_vectors[:,i]) for i in range(N)])]

for step in range(1, n_steps+1):
    grads1, psi1 = grads_mixture_entangled_fast(theta)
    k1 = mc_rhs_from_grads(grads1, psi1)
    th2 = theta + 0.5*dt*k1
    grads2, psi2 = grads_mixture_entangled_fast(th2)
    k2 = mc_rhs_from_grads(grads2, psi2)
    th3 = theta + 0.5*dt*k2
    grads3, psi3 = grads_mixture_entangled_fast(th3)
    k3 = mc_rhs_from_grads(grads3, psi3)
    th4 = theta + dt*k3
    grads4, psi4 = grads_mixture_entangled_fast(th4)
    k4 = mc_rhs_from_grads(grads4, psi4)
    theta = theta + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    if step % record_every == 0:
        _, _, psi_now, _, _, _, _ = assemble_joint_phi_entangled_fast(theta)
        exps_m.append(np.array([np.sum(np.abs(psi_now)**2 * pos_vectors[:,i]) for i in range(N)]))
        times_m.append(step*dt)

exps_m = np.array(exps_m); times_m = np.array(times_m)

# Compute RMSE
def compute_rmse(A,B): return np.sqrt(np.mean((A-B)**2, axis=0))
rmse_mix_ent = compute_rmse(exps_m, exps_t)

print("RMSE mixture+entangling vs trotter:", np.round(rmse_mix_ent,6))
print("Elapsed time: {:.1f}s".format(time.time()-start_time))

# Plot results
plt.figure(figsize=(10,5))
colors = ['C0','C1','C2']
for i in range(N):
    plt.plot(times_t, exps_t[:,i], '-', color=colors[i], label=f'Trotter body{i+1}')
    plt.plot(times_m, exps_m[:,i], '--', color=colors[i], label=f'Mix+Ent body{i+1}')
plt.xlabel('time'); plt.ylabel('<x>')
plt.title(f'Mixture+Entangling variational vs Trotter (dt={dt}, T={total_time}), RMSE={np.round(rmse_mix_ent,4)}')
plt.legend(); plt.grid(True); plt.tight_layout()
plt.show()

# Save CSVs
outdir = Path("./data"); outdir.mkdir(parents=True, exist_ok=True)
pd.DataFrame({'time': times_t, 'x1': exps_t[:,0], 'x2': exps_t[:,1], 'x3': exps_t[:,2]}).to_csv(outdir/'three_body_trotter_expectations_short.csv', index=False)
pd.DataFrame({'time': times_m, 'x1': exps_m[:,0], 'x2': exps_m[:,1], 'x3': exps_m[:,2]}).to_csv(outdir/'three_body_mixture_entangling_expectations_short.csv', index=False)

print("Saved CSVs to /mnt/data/")
