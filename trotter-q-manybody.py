# Fixed version: remove placeholder and run full simulation.
import numpy as np
import math
import itertools
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# --- Parameters ---
N = 3
m = 3
d = 2**m
x_min, x_max = -4.0, 4.0
dx = (x_max - x_min) / d
hbar_eff = 1.0
masses = [1.0, 1.0, 1.0]
G = 1.0
epsilon = 0.5
dt = 1e-2
total_time = 1.0
n_steps = int(total_time / dt)

x_grid = np.array([x_min + k * dx for k in range(d)])

# Single-register operators
X = np.diag(x_grid)
omega = np.exp(-2j * np.pi / d)
F = np.array([[omega**(q*k) for k in range(d)] for q in range(d)], dtype=complex) / np.sqrt(d)
p_vals = (2 * np.pi * hbar_eff) / (d * dx) * (np.arange(d) - d/2)
P = F @ np.diag(p_vals) @ F.conj().T
P2 = P @ P
T_single_op = P2 / (2.0 * masses[0])

np.set_printoptions(precision=4, suppress=True)
print("Single-register position operator X ({}x{}):\n".format(d, d), X)
print("\nSingle-register momentum operator P ({}x{}):\n".format(d, d), np.round(P, 6))

# Joint-space dimension
D = d**N

def idx_to_multi(idx, base=d, length=N):
    out = []
    for _ in range(length):
        out.append(idx % base)
        idx //= base
    return tuple(out)  # (k1, k2, k3) least-significant first

# Potential diagonal
V_diag = np.zeros(D, dtype=float)
for idx in range(D):
    ks = idx_to_multi(idx)
    Vsum = 0.0
    for i, j in itertools.combinations(range(N), 2):
        xi = x_grid[ks[i]]
        xj = x_grid[ks[j]]
        r = abs(xi - xj)
        Vsum += -G * masses[i] * masses[j] / math.sqrt(r**2 + epsilon**2)
    V_diag[idx] = Vsum

# Kinetic diagonal in joint momentum basis
T_mom_diag = np.zeros(D, dtype=float)
for idx in range(D):
    qs = idx_to_multi(idx)
    Tsum = 0.0
    for i in range(N):
        pq = p_vals[qs[i]]
        Tsum += (pq**2) / (2.0 * masses[i])
    T_mom_diag[idx] = Tsum

# Full QFT on joint space
F_total = F
for _ in range(N-1):
    F_total = np.kron(F_total, F)

# Define Trotter step
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

# Initial product state (Gaussians)
def gaussian_amplitudes(center, sigma=0.7):
    amp = np.exp(-0.5 * ((x_grid - center) / sigma)**2)
    amp = amp / np.linalg.norm(amp)
    return amp.astype(complex)

centers = [-2.0, 0.0, 2.0]
psis = [gaussian_amplitudes(c, sigma=0.7) for c in centers]
psi0 = psis[0]
for j in range(1, N):
    psi0 = np.kron(psi0, psis[j])
psi0 = psi0 / np.linalg.norm(psi0)

def compute_expectations(state):
    probs = np.abs(state)**2
    exps = []
    for i in range(N):
        exp_i = 0.0
        for idx in range(D):
            ks = idx_to_multi(idx)
            exp_i += x_grid[ks[i]] * probs[idx]
        exps.append(exp_i)
    return exps

# Simulation loop
psi = psi0.copy()
history = []
history.append((0.0, compute_expectations(psi)))

for step in range(1, n_steps+1):
    psi = trotter_step(psi, dt)
    t = step * dt
    history.append((t, compute_expectations(psi)))

# Save results
times = [h[0] for h in history]
exps = np.array([h[1] for h in history])
df = pd.DataFrame({'time': times, 'x1': exps[:,0], 'x2': exps[:,1], 'x3': exps[:,2]})
outdir = Path("./data"); outdir.mkdir(parents=True, exist_ok=True)
csv_path = outdir / "three_body_qrnn_worked_example_expectations.csv"
df.to_csv(csv_path, index=False)

# Plot
plt.figure(figsize=(8,4.5))
plt.plot(df['time'], df['x1'], label='x1')
plt.plot(df['time'], df['x2'], label='x2')
plt.plot(df['time'], df['x3'], label='x3')
plt.xlabel('time')
plt.ylabel('<x>')
plt.title('Expectations <x_i(t)> for 1D 3-body worked example (d=8)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Replace these lines that caused the error:
# from ace_tools import display_dataframe_to_user
# display_dataframe_to_user("three_body_expectations", df.head(20))

# --- With these lines (works in any Jupyter/Python environment) ---
from IPython.display import display
print("First 20 rows of results:")
display(df.head(20))

# Optionally print simple summary
print("\nSummary statistics:")
print(df.describe())

# Save the plot to a PNG file (if you want to download it)
plt.savefig('./data/three_body_expectations.png', dpi=150, bbox_inches='tight')
print("\nSaved plot to: ./data/three_body_expectations.png")

# The CSV was already saved; you can also re-save or inspect it:
print("Saved CSV to:", './data/three_body_qrnn_worked_example_expectations.csv')


print(f"\nSaved expectations CSV to: {csv_path}")
