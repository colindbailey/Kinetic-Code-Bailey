#Code Developed By Colin Bailey

import numpy as np
import matplotlib.pyplot as plt

# =========================
# Simulation setup (unchanged)
# =========================

# Constants
R = 8.314               # J/(mol·K)
T = 84 + 273.15         # K

# Optimized Arrhenius parameters
arrhenius_params = {
    'k_AE': {'A': 2.111166e+07, 'E_a': 52574.77},
    'k_AC': {'A': 6.544178e+04, 'E_a': 36818.67},
    'k_EI': {'A': 2.054356e+07, 'E_a': 46486.44},
    'k_CF': {'A': 3.595747e+07, 'E_a': 44463.55},
    'k_CB': {'A': 1.354610e+06, 'E_a': 36754.73},
    'k_FH': {'A': 2.989109e+06, 'E_a': 49925.99},
    'k_IH': {'A': 6.008348e+06, 'E_a': 58042.22},
    'k_HF': {'A': 2.346839e+07, 'E_a': 50716.43},
    'k_form': 4.098901              # mol/(L·s)
}

bimolecular_reactions = ['k_AE','k_AC','k_CF','k_FH','k_EI','k_IH']

# Compute k(T) in SI
k_values = {}
for key, p in arrhenius_params.items():
    if key == 'k_form':
        # mol/(L·s) -> mol/(m^3·s)
        k_values[key] = p * 1e3
    else:
        A_i, E_a = p['A'], p['E_a']
        # L/(mol·s) -> m^3/(mol·s) for bimolecular steps
        A_si = A_i * 1e-3 if key in bimolecular_reactions else A_i
        k_values[key] = A_si * np.exp(-E_a/(R*T))
    if key != 'k_form':
        unit = 'm³/(mol·s)' if key in bimolecular_reactions else '1/s'
        print(f"{key}: k = {k_values[key]:.6e} {unit}")

# Reactor & flow geometry
reactor_volume   = 30e-6        # m³
D_tube           = 0.0016       # m
Ac               = np.pi*(D_tube/2)**2
length           = reactor_volume / Ac
flow_rate_mL_min = 1.76         # mL/min (simulation)
Q                = flow_rate_mL_min * 1e-6 / 60  # m³/s
u                = Q / Ac       # m/s   (superficial velocity)
D_axial          = 1e-5         # m²/s

# Discretization
N  = 1000
dx = length / (N-1)
x  = np.linspace(0, length, N)

# Initial concentrations (mol/m³)
initial_concentration_A   = 175.0    # 0.305 mol/L → 305 mol/m³
initial_concentration_IPA = 4100.0   # 4.1 mol/L → 4100 mol/m³

C_A   = np.full(N, initial_concentration_A)
C_IPA = np.full(N, initial_concentration_IPA)
C_B   = np.zeros(N)
C_C   = np.zeros(N)
C_E   = np.zeros(N)
C_F   = np.zeros(N)
C_H   = np.zeros(N)
C_I   = np.zeros(N)
C_X   = np.zeros(N)

# Track A consumed
A_consumed = np.zeros(N)

# Time‐stepping
dt                     = 0.01
steady_state_tolerance = 1e-6
max_iterations         = 5_000_000
f_X                    = dt  # factor for carbocation formation

# Main loop
for iteration in range(1, max_iterations+1):
    # Save for convergence
    A_prev, IPA_prev = C_A.copy(), C_IPA.copy()
    B_prev,C_prev   = C_B.copy(), C_C.copy()
    E_prev,F_prev   = C_E.copy(), C_F.copy()
    H_prev,I_prev,X_prev = C_H.copy(), C_I.copy(), C_X.copy()

    # Raw rates
    r_form = k_values['k_form'] * C_IPA[1:-1]
    r_AE   = k_values['k_AE'] * C_A[1:-1]*C_X[1:-1]
    r_AC   = k_values['k_AC'] * C_A[1:-1]*C_X[1:-1]
    r_CF   = k_values['k_CF'] * C_C[1:-1]*C_X[1:-1]
    r_CB   = k_values['k_CB'] * C_C[1:-1]
    r_FH   = k_values['k_FH'] * C_F[1:-1]*C_X[1:-1]
    r_EI   = k_values['k_EI'] * C_E[1:-1]*C_X[1:-1]
    r_IH   = k_values['k_IH'] * C_I[1:-1]*C_X[1:-1]
    r_HF   = k_values['k_HF'] * C_H[1:-1]

    # Update X
    dX = f_X*r_form - f_X*(r_AE+r_AC+r_CF+r_FH+r_EI+r_IH)
    C_X[1:-1] += dt * (
        -u*(C_X[1:-1]-C_X[:-2])/dx
        + D_axial*(C_X[2:]-2*C_X[1:-1]+C_X[:-2])/dx**2
        + dX
    )
    C_X = np.maximum(C_X, 0)

    # Update IPA
    C_IPA[1:-1] += dt * (
        -u*(C_IPA[1:-1]-C_IPA[:-2])/dx
        + D_axial*(C_IPA[2:]-2*C_IPA[1:-1]+C_IPA[:-2])/dx**2
        - f_X*r_form
    )
    C_IPA = np.maximum(C_IPA, 0)

    # Update A & consumption
    delta_AE = r_AE
    delta_AC = r_AC
    C_A[1:-1] += dt * (
        -u*(C_A[1:-1]-C_A[:-2])/dx
        + D_axial*(C_A[2:]-2*C_A[1:-1]+C_A[:-2])/dx**2
        - delta_AE - delta_AC
    )
    A_consumed[1:-1] += delta_AE + delta_AC

    # Update C
    C_C[1:-1] += dt * (
        -u*(C_C[1:-1]-C_C[:-2])/dx
        + D_axial*(C_C[2:]-2*C_C[1:-1]+C_C[:-2])/dx**2
        + delta_AC - r_CF - r_CB
    )
    C_C[1:-1] = np.minimum(C_C[1:-1], A_consumed[1:-1])

    # Update E
    C_E[1:-1] += dt * (
        -u*(C_E[1:-1]-C_E[:-2])/dx
        + D_axial*(C_E[2:]-2*C_E[1:-1]+C_E[:-2])/dx**2
        + delta_AE - r_EI
    )
    C_E[1:-1] = np.minimum(C_E[1:-1], A_consumed[1:-1])

    # Update F
    C_F[1:-1] += dt * (
        -u*(C_F[1:-1]-C_F[:-2])/dx
        + D_axial*(C_F[2:]-2*C_F[1:-1]+C_F[:-2])/dx**2
        + r_CF - r_FH + r_HF
    )
    C_F[1:-1] = np.minimum(C_F[1:-1], A_consumed[1:-1])

    # Update B
    C_B[1:-1] += dt * (
        -u*(C_B[1:-1]-C_B[:-2])/dx
        + D_axial*(C_B[2:]-2*C_B[1:-1]+C_B[:-2])/dx**2
        + r_CB
    )
    C_B[1:-1] = np.minimum(C_B[1:-1], A_consumed[1:-1])

    # Update H
    C_H[1:-1] += dt * (
        -u*(C_H[1:-1]-C_H[:-2])/dx
        + D_axial*(C_H[2:]-2*C_H[1:-1]+C_H[:-2])/dx**2
        + r_IH + r_FH - r_HF
    )
    C_H[1:-1] = np.minimum(C_H[1:-1], A_consumed[1:-1])

    # Update I
    C_I[1:-1] += dt * (
        -u*(C_I[1:-1]-C_I[:-2])/dx
        + D_axial*(C_I[2:]-2*C_I[1:-1]+C_I[:-2])/dx**2
        + r_EI - r_IH
    )
    C_I[1:-1] = np.minimum(C_I[1:-1], A_consumed[1:-1])

    # Boundary conditions
    C_A[[0,-1]]   = initial_concentration_A,   C_A[-2]
    C_IPA[[0,-1]] = initial_concentration_IPA, C_IPA[-2]
    for arr in (C_C, C_E, C_F, C_B, C_H, C_I):
        arr[[0,-1]] = 0.0, arr[-2]

    # Convergence check
    max_change = max(
        np.max(np.abs(C_A  - A_prev)),
        np.max(np.abs(C_IPA-IPA_prev)),
        np.max(np.abs(C_B  - B_prev)),
        np.max(np.abs(C_C  - C_prev)),
        np.max(np.abs(C_E  - E_prev)),
        np.max(np.abs(C_F  - F_prev)),
        np.max(np.abs(C_H  - H_prev)),
        np.max(np.abs(C_I  - I_prev)),
        np.max(np.abs(C_X  - X_prev))
    )
    if max_change < steady_state_tolerance:
        print(f"Steady state reached after {iteration} iterations.")
        break

# Print yields
initial_moles_A = initial_concentration_A * reactor_volume
for sp, arr in zip(['A','B','C','E','F','H','I'],
                   [C_A,C_B,C_C,C_E,C_F,C_H,C_I]):
    y = (arr[-1]*reactor_volume)/initial_moles_A*100
    print(f"Yield of {sp}: {y:.2f}%")

# =========================
# OVERLAY: paste rows (time) → x using SIMULATION velocity u
# =========================
# Row format (flow column is ignored):
# 0=TempC, 1=flow_mL_min(ignored), 2=HBA_eq, 3=IPA(mol/L), 4=time_s,
# 5=A, 6=B, 7=C, 8=D, 9=E, 10=F, 11=G, 12=H, 13=L, 14=J

PASTED_DATA = """
84	3.982044594	0.9	2.389226756	0	0.181002027	0	0	0	0	0	0	0	0	0
84	3.982044594	0.9	2.333855598	180	0.120194263	0.000723962	0.008302856	0.00388944	0.011111147	0.012035563	0.003720689	0.003128323	0.000746018	0.00165056
84	3.982044594	0.9	2.286332309	300	0.067590006	0.001199544	0.015675094	0.007185374	0.021281258	0.017813142	0.009704089	0.008159113	0.001138211	0.003049253
84	3.982044594	0.9	2.256657185	450	0.044353021	0.001443154	0.027588667	0.004152277	0.013950754	0.02316054	0.014421972	0.012125869	0.001118489	0.001762099
84	3.982044594	0.9	2.243247818	600	0.031310809	0.001631359	0.031031914	0.003409191	0.009046801	0.02859117	0.015695508	0.013196647	0.001635145	0.001446757
84	3.982044594	0.9	2.222771419	900	0.003864521	0.00335695	0.010831544	0.002504262	0.000227147	0.072455899	0.002375966	0.001997691	0.003912814	0.001062732
84	3.982044594	0.9	2.236840847	1200	0.007033957	0.002186845	0.020503626	0.00260666	0.001362367	0.051424967	0.008494379	0.007142	0.005665383	0.001106187
""".strip()

def _parse_rows(text):
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.replace(",", " ").split()
        rows.append([float(p) for p in parts])
    return np.array(rows, float)

def _to_m3(conc_mol_per_L):
    return conc_mol_per_L * 1000.0  # mol/L -> mol/m^3

# ---------- Overlay builder: time (s) -> x (m) using simulation velocity u ----------
def build_overlay_using_sim_u(pasted_text, u_sim, L):
    tbl = _parse_rows(pasted_text)
    time_s = tbl[:, 4]                  # seconds
    x_exp  = u_sim * time_s             # m (use simulation velocity)

    # DEBUG: show mapping (optional; comment out later)
    for t_i, x_i in zip(time_s, x_exp):
        print(f"[map] t={t_i:.3f} s -> x={x_i:.6f} m  (L={L:.6f} m)")

    # Keep ONLY points that lie inside reactor length, with tiny tolerance
    eps = 1e-9 * max(1.0, L)
    mask = x_exp <= (L + eps)
    x_exp = x_exp[mask]
    tbl   = tbl[mask]

    # Convert mol/L -> mol/m^3 and collect species we model
    data = {}
    data["IPA"] = _to_m3(tbl[:, 3])
    colmap = {"A":5, "B":6, "C":7, "E":9, "F":10, "H":12}
    for sp, idx in colmap.items():
        data[sp] = _to_m3(tbl[:, idx])

    # Sort by x so lines connect left->right
    order = np.argsort(x_exp)
    x_exp = x_exp[order]
    for k in data:
        data[k] = np.asarray(data[k])[order]
    return x_exp, data

# ---------- Build overlay from the PASTED_DATA block above ----------
x_exp, data_exp = build_overlay_using_sim_u(PASTED_DATA, u, length)
print(f"[Overlay] {len(x_exp)} points loaded (inside reactor with tolerance).")

# ---------- Plot: IPA & X with overlay (lines + points) ----------
plt.figure(figsize=(10,6))
ipa_line, = plt.plot(x, C_IPA, label='IPA (model)')
plt.plot(x, C_X,            label='X (model)')

if "IPA" in data_exp and len(x_exp):
    plt.plot(x_exp, data_exp["IPA"], linestyle='--', linewidth=1,
             color=ipa_line.get_color(), label='IPA (exp line)')
    plt.scatter(x_exp, data_exp["IPA"], s=36, marker='o',
                facecolor='none', edgecolor=ipa_line.get_color(),
                label='IPA (exp pts)')

plt.xlabel('Reactor Length (m)')
plt.ylabel('Concentration (mol/m³)')
plt.title('Concentration Profiles of IPA and X Along the Reactor')
plt.legend(ncol=2)
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------- Plot: A..I with overlay (lines + points) ----------
# Nice labels for each species code
nice = {
    'A': '4-HBA',
    'B': 'Sulfonated Impurity',
    'C': 'Mono-Alkylated',
    'E': '4-HBA Ester',
    'F': 'DIPBA',
    'H': 'DIPBA Ester',
    'I': 'Mono-Alkylated Ester',
}

plt.figure(figsize=(12,7))
line_handles = {}

# Plot model curves with nice labels
line_handles['A'], = plt.plot(x, C_A, label=f"{nice['A']} (model)")
line_handles['B'], = plt.plot(x, C_B, label=f"{nice['B']} (model)")
line_handles['C'], = plt.plot(x, C_C, label=f"{nice['C']} (model)")
line_handles['E'], = plt.plot(x, C_E, label=f"{nice['E']} (model)")
line_handles['F'], = plt.plot(x, C_F, label=f"{nice['F']} (model)")
line_handles['H'], = plt.plot(x, C_H, label=f"{nice['H']} (model)")
plt.plot(x, C_I, label=f"{nice['I']} (model)")  # no experimental I overlay

# Marker shapes keyed by species code
marker_map = {'A':'o', 'B':'v', 'C':'^', 'E':'<', 'F':'>', 'H':'s'}

# Overlay experimental lines + points, color-matched to model
if len(x_exp):
    for sp in ['A','B','C','E','F','H']:
        if sp in data_exp:
            color = line_handles[sp].get_color()
            # Plot line without legend entry
            plt.plot(x_exp, data_exp[sp], linestyle='--', linewidth=1,
                     color=color)
            # Scatter with legend label (so only 1 entry per species)
            plt.scatter(x_exp, data_exp[sp], s=40, marker=marker_map[sp],
                        facecolor='none', edgecolor=color,
                        label=f"{nice[sp]} (exp)")


plt.xlabel('Reactor Length (m)')
plt.ylabel('Concentration (mol/m³)')
plt.title('Species Profiles Along Reactor')
plt.legend(ncol=2)
plt.grid(True)
plt.tight_layout()
plt.show()

