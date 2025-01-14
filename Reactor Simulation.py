import numpy as np
import matplotlib.pyplot as plt

# Constants
R = 8.314  # Universal gas constant, J/(mol·K)
T = 80 + 273.15  # Temperature, K

# Updated Arrhenius parameters with provided optimized values from kinetic model_V23
arrhenius_params = {
    'k_AE': {'A': 2.111166e+07, 'E_a': 52574.77},  # A + X -> E
    'k_AC': {'A': 6.544178e+04, 'E_a': 36818.67},  # A + X -> C
    'k_EI': {'A': 2.054356e+07, 'E_a': 46486.44},  # E + X -> I
    'k_CF': {'A': 3.595747e+07, 'E_a': 44463.55},  # C + X -> F
    'k_CB': {'A': 1.354610e+06, 'E_a': 36754.73},  # C -> B
    'k_FH': {'A': 2.989109e+06, 'E_a': 49925.99},  # F + X -> H
    'k_IH': {'A': 6.008348e+06, 'E_a': 58042.22},  # I + X -> H
    'k_HF': {'A': 2.346839e+07, 'E_a': 50716.43},  # H -> F
    'k_form': 4.098901  # Formation rate constant of X
}

# Reactions involving X (bimolecular reactions)
bimolecular_reactions = ['k_AE', 'k_AC', 'k_CF', 'k_FH']

# Convert rate constants to SI units and calculate k_values at temperature T
k_values = {}
for key, params in arrhenius_params.items():
    if key == 'k_form':
        k_values[key] = params  # k_form is already a constant in mol/(L·s)
        continue

    A_i = params['A']
    E_a_i = params['E_a']
    if key in bimolecular_reactions:
        A_i_SI = A_i * 1e-3  # Convert L/(mol·s) to m³/(mol·s)
        units = 'm³/(mol·s)'
    else:
        A_i_SI = A_i  # 1/s
        units = '1/s'
    k_i = A_i_SI * np.exp(-E_a_i / (R * T))
    k_values[key] = k_i
    print(f"{key}: k = {k_i:.6e} {units}")

# Reactor parameters
reactor_volume = 30e-6  # Reactor volume in m³ (30 mL)
D_tube = 0.0016  # Tube diameter in meters (1.6 mm)
Ac = np.pi * (D_tube / 2)**2  # Cross-sectional area in m²
length = reactor_volume / Ac  # Reactor length in meters
flow_rate_mL_min = 2.0  # Flow rate in mL/min
Q = flow_rate_mL_min * 1e-6 / 60  # m³/s
u = Q / Ac  # Linear velocity in m/s
D_axial = 1e-5  # m²/s

# Spatial discretization
N = 1000  # Number of spatial grid points
dx = length / (N - 1)
x = np.linspace(0, length, N)

# Initial concentrations (mol/m³)
initial_concentration_A = 0.305 * 1e3  # A (converted to mol/m³)
initial_concentration_IPA = 4.1 * 1e3  # IPA (converted to mol/m³)
C_A = np.full(N, initial_concentration_A)    # A
C_IPA = np.full(N, initial_concentration_IPA)  # IPA
C_B = np.zeros(N)      # B
C_C = np.zeros(N)      # C
C_E = np.zeros(N)      # E
C_F = np.zeros(N)      # F
C_H = np.zeros(N)      # H
C_I = np.zeros(N)      # I
C_X = np.zeros(N)      # X

# Track the amount of A that has been consumed to limit product formation
A_consumed = np.zeros(N)

# Time-stepping parameters
dt = 0.01  # Time step (s)
max_iterations = 5000000
steady_state_tolerance = 1e-6

# Reaction rate calculation function
def reaction_rates(C_A, C_C, C_F, C_E, C_I, C_H, C_X):
    r_AE = k_values['k_AE'] * C_A * C_X
    r_AC = k_values['k_AC'] * C_A * C_X
    r_CF = k_values['k_CF'] * C_C * C_X
    r_CB = k_values['k_CB'] * C_C
    r_FH = k_values['k_FH'] * C_F * C_X
    r_EI = k_values['k_EI'] * C_E * C_X
    r_IH = k_values['k_IH'] * C_I * C_X
    r_HF = k_values['k_HF'] * C_H
    return r_AE, r_AC, r_CF, r_CB, r_FH, r_EI, r_IH, r_HF

# Time-stepping loop
for iteration in range(1, max_iterations + 1):
    C_A_prev, C_IPA_prev = C_A.copy(), C_IPA.copy()
    C_B_prev, C_C_prev = C_B.copy(), C_C.copy()
    C_E_prev, C_F_prev = C_E.copy(), C_F.copy()
    C_H_prev, C_I_prev, C_X_prev = C_H.copy(), C_I.copy(), C_X.copy()

    # Formation rate of X depends on IPA concentration
    r_form_X = k_values['k_form'] * C_IPA[1:-1]

    # Calculate reaction rates
    r_AE, r_AC, r_CF, r_CB, r_FH, r_EI, r_IH, r_HF = reaction_rates(
        C_A[1:-1], C_C[1:-1], C_F[1:-1], C_E[1:-1], C_I[1:-1], C_H[1:-1], C_X[1:-1]
    )

    # Update equations for all species with consideration for mass conservation

    # Update X based on formation from IPA and consumption in other reactions
    dX_formation = r_form_X * dt
    dX_consumed = (r_AE + r_AC + r_CF + r_FH + r_EI + r_IH) * dt
    dX = dX_formation - dX_consumed
    C_X[1:-1] += dt * (-u * (C_X[1:-1] - C_X[:-2]) / dx +
                       D_axial * (C_X[2:] - 2 * C_X[1:-1] + C_X[:-2]) / dx**2 + dX)
    C_X[1:-1] = np.maximum(C_X[1:-1], 0)  # Ensure X concentration doesn't go negative

    # Update IPA concentration based on formation of X
    C_IPA[1:-1] += dt * (-u * (C_IPA[1:-1] - C_IPA[:-2]) / dx +
                         D_axial * (C_IPA[2:] - 2 * C_IPA[1:-1] + C_IPA[:-2]) / dx**2 - r_form_X * dt)
    C_IPA[1:-1] = np.maximum(C_IPA[1:-1], 0)

    # Update A concentration and track how much A is consumed
    delta_AE = r_AE * dt
    delta_AC = r_AC * dt
    C_A[1:-1] += dt * (-u * (C_A[1:-1] - C_A[:-2]) / dx +
                       D_axial * (C_A[2:] - 2 * C_A[1:-1] + C_A[:-2]) / dx**2 - delta_AE - delta_AC)
    A_consumed[1:-1] += delta_AE + delta_AC

    # Ensure that product formation is limited by the amount of A consumed
    C_C[1:-1] += dt * (-u * (C_C[1:-1] - C_C[:-2]) / dx +
                       D_axial * (C_C[2:] - 2 * C_C[1:-1] + C_C[:-2]) / dx**2 + delta_AC - r_CF * dt - r_CB * dt)
    C_C[1:-1] = np.minimum(C_C[1:-1], A_consumed[1:-1])

    C_E[1:-1] += dt * (-u * (C_E[1:-1] - C_E[:-2]) / dx +
                       D_axial * (C_E[2:] - 2 * C_E[1:-1] + C_E[:-2]) / dx**2 + delta_AE - r_EI * dt)
    C_E[1:-1] = np.minimum(C_E[1:-1], A_consumed[1:-1])

    C_F[1:-1] += dt * (-u * (C_F[1:-1] - C_F[:-2]) / dx +
                       D_axial * (C_F[2:] - 2 * C_F[1:-1] + C_F[:-2]) / dx**2 + r_CF * dt - r_FH * dt + r_HF * dt)
    C_F[1:-1] = np.minimum(C_F[1:-1], A_consumed[1:-1])

    C_B[1:-1] += dt * (-u * (C_B[1:-1] - C_B[:-2]) / dx +
                       D_axial * (C_B[2:] - 2 * C_B[1:-1] + C_B[:-2]) / dx**2 + r_CB * dt)
    C_B[1:-1] = np.minimum(C_B[1:-1], A_consumed[1:-1])

    C_H[1:-1] += dt * (-u * (C_H[1:-1] - C_H[:-2]) / dx +
                       D_axial * (C_H[2:] - 2 * C_H[1:-1] + C_H[:-2]) / dx**2 + r_IH * dt + r_FH * dt - r_HF * dt)
    C_H[1:-1] = np.minimum(C_H[1:-1], A_consumed[1:-1])

    C_I[1:-1] += dt * (-u * (C_I[1:-1] - C_I[:-2]) / dx +
                       D_axial * (C_I[2:] - 2 * C_I[1:-1] + C_I[:-2]) / dx**2 + r_EI * dt - r_IH * dt)
    C_I[1:-1] = np.minimum(C_I[1:-1], A_consumed[1:-1])

    # Apply boundary conditions
    C_IPA[0], C_IPA[-1] = initial_concentration_IPA, C_IPA[-2]
    C_X[0], C_X[-1] = 0.0, C_X[-2]
    C_A[0], C_A[-1] = initial_concentration_A, C_A[-2]
    for conc in [C_C, C_F, C_B, C_E, C_H, C_I]:
        conc[0], conc[-1] = 0.0, conc[-2]

    # Ensure concentrations remain non-negative
    for conc in [C_IPA, C_X, C_A, C_B, C_C, C_E, C_F, C_H, C_I]:
        np.maximum(conc, 0, out=conc)

    # Check for steady-state convergence
    max_change = max(
        np.max(np.abs(C_IPA - C_IPA_prev)),
        np.max(np.abs(C_A - C_A_prev)),
        np.max(np.abs(C_B - C_B_prev)),
        np.max(np.abs(C_C - C_C_prev)),
        np.max(np.abs(C_E - C_E_prev)),
        np.max(np.abs(C_F - C_F_prev)),
        np.max(np.abs(C_H - C_H_prev)),
        np.max(np.abs(C_I - C_I_prev)),
        np.max(np.abs(C_X - C_X_prev))
    )
    if max_change < steady_state_tolerance:
        print(f"Steady state reached after {iteration} iterations.")
        break

# Calculate yields for all compounds
initial_moles_A = initial_concentration_A * reactor_volume  # Initial moles of A

# Final moles of each compound
final_moles = {
    'A': C_A[-1] * reactor_volume,
    'B': C_B[-1] * reactor_volume,
    'C': C_C[-1] * reactor_volume,
    'E': C_E[-1] * reactor_volume,
    'F': C_F[-1] * reactor_volume,
    'H': C_H[-1] * reactor_volume
}

# Calculate and print yields
yields = {compound: (final_moles[compound] / initial_moles_A) * 100 for compound in final_moles}

for compound, yield_value in yields.items():
    print(f"Yield of {compound}: {yield_value:.2f}%")

# Plot the final concentrations
plt.figure(figsize=(12, 8))
plt.plot(x, C_A, label='A')
plt.plot(x, C_C, label='C')
plt.plot(x, C_E, label='E')
plt.plot(x, C_F, label='F')
plt.plot(x, C_B, label='B')
plt.plot(x, C_H, label='H')
plt.plot(x, C_I, label='I')
plt.xlabel('Reactor Length (m)', fontsize=14)
plt.ylabel('Concentration (mol/m³)', fontsize=14)
plt.title('Concentration Profiles Along the Reactor', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot total mass to check mass balance
total_mass = C_A + C_C + C_E + C_F + C_B + C_H + C_I
plt.figure(figsize=(12, 6))
plt.plot(x, total_mass, label='Total Mass', color='green')
plt.xlabel('Reactor Length (m)', fontsize=14)
plt.ylabel('Total Concentration (mol/m³)', fontsize=14)
plt.title('Total Mass Along the Reactor', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot IPA and X separately
plt.figure(figsize=(12, 6))
plt.plot(x, C_IPA, label='IPA', color='blue')
plt.plot(x, C_X, label='X', color='red')
plt.xlabel('Reactor Length (m)', fontsize=14)
plt.ylabel('Concentration (mol/m³)', fontsize=14)
plt.title('Concentration Profiles of IPA and X Along the Reactor', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()
