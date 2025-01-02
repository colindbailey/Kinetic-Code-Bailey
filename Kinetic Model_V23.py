import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

# Load concentration and time data from CSV for multiple experiments and compounds
def load_concentration_data(file_path):
    df = pd.read_csv(file_path)
    experiments = df['experiment'].unique()
    compounds = ['A', 'B', 'C', 'E', 'F', 'H', 'IPA']  # Added 'IPA' as it's needed for `X_ss` calculation
    all_data = {compound: {'time': {}, 'concentration': {}} for compound in compounds}
    temperatures = {}

    for exp in experiments:
        exp_data = df[df['experiment'] == exp]
        time = exp_data['time'].values  # Time in seconds
        temp = exp_data['Temp'].iloc[0]
        temperatures[exp] = temp + 273.15  # Convert to Kelvin
        for compound in compounds:
            conc_data = exp_data[compound].values
            all_data[compound]['time'][exp] = time
            all_data[compound]['concentration'][exp] = conc_data

    return all_data, compounds, experiments, temperatures

# Updated reaction network:
# All reactions except C -> B are dependent on X_ss
# Added reversible reaction F ⇌ H
def reaction_network(t, y, arrhenius_params, temperature, k_form):
    R = 8.314  # J/(mol·K), gas constant
    A_values = arrhenius_params[::2]  # Pre-exponential factors
    E_values = arrhenius_params[1::2]  # Activation energies
    k_values = A_values * np.exp(-E_values / (R * temperature))  # Rate constants

    # Unpack rate constants
    k_AE, k_AC, k_EI, k_CF, k_CB, k_FH, k_IH, k_HF = k_values

    # Species concentrations
    A, B, C, E, F, H, I, IPA = y

    # Calculate X_ss as a function of IPA concentration
    X_ss = k_form * IPA

    # Define effective rate constants by incorporating X_ss where applicable
    k_AE_eff = k_AE * X_ss  # A + X -> E
    k_AC_eff = k_AC * X_ss  # A + X -> C
    k_EI_eff = k_EI * X_ss  # E + X -> I
    k_IH_eff = k_IH * X_ss  # I + X -> H
    k_CF_eff = k_CF * X_ss  # C + X -> F
    k_FH_eff = k_FH * X_ss  # F + X -> H
             
    

    # Define reaction rates
    r_AE = k_AE_eff * A        # A + X -> E
    r_AC = k_AC_eff * A        # A + X -> C
    r_EI = k_EI_eff * E        # E + X -> I
    r_IH = k_IH_eff * I        # I + X -> H
    r_CF = k_CF_eff * C        # C + X -> F
    r_CB = k_CB * C            # C -> B (independent of X_ss)
    r_FH = k_FH_eff * F        # F + X -> H
    r_HF = k_HF * H            # H -> F
    

    # Differential equations
    dA_dt = -r_AE - r_AC
    dB_dt = r_CB
    dC_dt = r_AC - r_CF - r_CB
    dE_dt = r_AE - r_EI
    dF_dt = r_CF - r_FH + r_HF
    dH_dt = r_IH + r_FH - r_HF
    dI_dt = r_EI - r_IH
    dIPA_dt = -k_form * IPA  # IPA is consumed to generate X

    return [dA_dt, dB_dt, dC_dt, dE_dt, dF_dt, dH_dt, dI_dt, dIPA_dt]

# Wrapper for solving ODEs for each experiment individually
def solve_kinetics_wrapper_for_experiment(t, arrhenius_params, initial_conditions, temperature, k_form):
    t_span = (t[0], t[-1])
    solution = solve_ivp(
        reaction_network, t_span, initial_conditions,
        args=(arrhenius_params, temperature, k_form), t_eval=t, method='BDF'
    )
    if not solution.success:
        raise RuntimeError(f"ODE solver failed: {solution.message}")
    return solution.y  # Return the entire concentration matrix

# Function to calculate mass balance
def calculate_mass_balance(solution, molecular_weights, species_indices):
    # Calculate total mass at each time point
    total_mass = np.zeros(solution.shape[1])
    for species_name, idx in species_indices.items():
        total_mass += solution[idx, :] * molecular_weights[species_name]
    return total_mass

# Main function to perform parameter estimation across all experiments
def main():
    file_path = 'C:/Users/baileycd2/Desktop/Kinetic Experiments_2_2.csv'
    all_data, compounds, experiments, temperatures = load_concentration_data(file_path)

    # Species list including IPA
    species = ['A', 'B', 'C', 'E', 'F', 'H', 'I', 'IPA']
    species_indices = {name: idx for idx, name in enumerate(species)}

    # Molecular weights (g/mol) - replace with actual values
    molecular_weights = {
        'A': 138.12, 'B': 260.26, 'C': 180.2, 'E': 180.2,
        'F': 222.28, 'H': 264.37, 'I': 222.28, 'IPA': 60.1
    }

    # Initial guesses for Arrhenius parameters and k_form
    initial_guess = []
    for _ in range(8):  # 8 reactions including k_HF
        initial_guess.extend([1e6, 50000])  # [A_i, E_a_i] for each rate constant
    initial_guess.append(1.5)  # k_form (mol/(L·s))

    # Bounds for the parameters
    bounds_lower = [0] * 17  # 8 reactions * 2 parameters each + k_form
    bounds_upper = [1e12, 1e5] * 8 + [10]  # Last parameter k_form bounded to [0, 10]

    combined_time_data = []
    initial_conditions_list = []
    experiments_list = []
    temperature_list = []

    for exp in experiments:
        exp_time_data = all_data['A']['time'][exp]
        # Set initial conditions for species
        initial_conditions = [
            all_data[comp]['concentration'][exp][0] if comp in compounds else 0.0 for comp in species
        ]
        combined_time_data.append(exp_time_data)
        initial_conditions_list.append(initial_conditions)
        experiments_list.append(exp)
        temperature_list.append(temperatures[exp])

    def residuals_function(params):
        arrhenius_params = params[:-1]
        k_form = params[-1]
        residuals = []
        for time_data, initial_conditions, exp, temperature in zip(
            combined_time_data, initial_conditions_list, experiments_list, temperature_list
        ):
            try:
                solution = solve_kinetics_wrapper_for_experiment(
                    time_data, arrhenius_params, initial_conditions, temperature, k_form
                )
            except RuntimeError as e:
                print(f"Experiment {exp} failed due to ODE solver failure: {e}")
                # Fill residuals with large values to penalize failed fits
                num_time_points = len(time_data)
                num_compounds = len(compounds)
                residuals.extend([1e6] * num_time_points * num_compounds)
                continue
            for compound in compounds:
                index = species_indices[compound]  # Correct index for the compound
                model_conc = solution[index, :]
                actual_conc = all_data[compound]['concentration'][exp]
                residuals.extend(actual_conc - model_conc)
        return np.array(residuals)

    # Perform least squares optimization
    result = least_squares(
        residuals_function, initial_guess, bounds=(bounds_lower, bounds_upper),
        max_nfev=20000, verbose=2  # Increased max_nfev for better convergence
    )

    k_labels = ['k_AE', 'k_AC', 'k_EI', 'k_CF', 'k_CB', 'k_FH', 'k_IH', 'k_HF', 'k_form']

    print("\nOptimized Arrhenius parameters (A_i and E_a):")
    for i, label in enumerate(k_labels[:-1]):
        A_i = result.x[2 * i]
        E_i = result.x[2 * i + 1]
        print(f"{label}: A = {A_i:.6e} L/(mol·s), E_a = {E_i:.2f} J/mol")
    print(f"k_form: {result.x[-1]:.6f} mol/(L·s)")

    residuals = result.fun
    ssr = np.sum(residuals**2)
    ss_total = np.sum((residuals - np.mean(residuals))**2)
    r_squared_total = 1 - (ssr / ss_total)
    print(f"\nSum of Squared Residuals (SSR): {ssr}")
    print(f"Overall Coefficient of Determination (R²): {r_squared_total:.3f}")

    for compound in compounds:
        actual_concentration_data = []
        fitted_concentration_data = []
        index = species_indices[compound]  # Correct index for the compound
        for time_data, initial_conditions, exp, temperature in zip(
            combined_time_data, initial_conditions_list, experiments_list, temperature_list
        ):
            try:
                actual_conc = all_data[compound]['concentration'][exp]
                solution = solve_kinetics_wrapper_for_experiment(
                    time_data, result.x[:-1], initial_conditions, temperature, result.x[-1]
                )
                model_conc = solution[index, :]
                actual_concentration_data.extend(actual_conc)
                fitted_concentration_data.extend(model_conc)
            except RuntimeError as e:
                print(f"Skipping experiment {exp} for compound {compound} due to ODE solver failure: {e}")
                continue
        actual_concentration_data = np.array(actual_concentration_data)
        fitted_concentration_data = np.array(fitted_concentration_data)
        residuals_compound = actual_concentration_data - fitted_concentration_data
        ssr_compound = np.sum(residuals_compound**2)
        ss_total_compound = np.sum((actual_concentration_data - np.mean(actual_concentration_data))**2)
        r_squared_compound = 1 - (ssr_compound / ss_total_compound)
        print(f"R² for {compound}: {r_squared_compound:.3f}")

    # Optional: Plot optimized vs actual concentrations for visual verification
    # Example for compound 'A'
    plt.figure(figsize=(10, 6))
    for exp in experiments:
        exp_time_data = all_data['A']['time'][exp]
        actual_conc = all_data['A']['concentration'][exp]
        initial_conditions = initial_conditions_list[experiments_list.index(exp)]
        try:
            solution = solve_kinetics_wrapper_for_experiment(
                exp_time_data, result.x[:-1], initial_conditions, temperature=temperatures[exp], k_form=result.x[-1]
            )
            model_conc = solution[index, :]
            plt.plot(exp_time_data, actual_conc, 'o', label=f'Actual A - Exp {exp}')
            plt.plot(exp_time_data, model_conc, '-', label=f'Fitted A - Exp {exp}')
        except RuntimeError as e:
            print(f"Skipping plotting for experiment {exp} due to ODE solver failure: {e}")
            continue
    plt.xlabel('Time (s)')
    plt.ylabel('Concentration of A (mol/L)')
    plt.title('Actual vs Fitted Concentration of A Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
