#Code Developed by Colin Bailey

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

# Load concentration and time data from CSV for multiple experiments and compounds
def load_concentration_data(file_path):
    df = pd.read_csv(file_path)
    experiments = df['experiment'].unique()
    compounds = ['A', 'B', 'C', 'E', 'F', 'H', 'IPA']  # 'IPA' is needed for X_ss calculation
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
    total_mass = np.zeros(solution.shape[1])
    for species_name, idx in species_indices.items():
        total_mass += solution[idx, :] * molecular_weights[species_name]
    return total_mass

# Main function to perform parameter estimation across all experiments
def main():
    file_path = 'C:/Users/baileycd2/Downloads/Kinetic Experiments.csv'
    all_data, compounds, experiments, temperatures = load_concentration_data(file_path)

    # Species list including IPA
    species = ['A', 'B', 'C', 'E', 'F', 'H', 'I', 'IPA']
    species_indices = {name: idx for idx, name in enumerate(species)}

    # Molecular weights (g/mol)
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
                num_time_points = len(time_data)
                num_compounds = len(compounds)
                residuals.extend([1e6] * num_time_points * num_compounds)
                continue
            for compound in compounds:
                index = species_indices[compound]
                model_conc = solution[index, :]
                actual_conc = all_data[compound]['concentration'][exp]
                residuals.extend(actual_conc - model_conc)
        return np.array(residuals)

    # Perform least squares optimization
    result = least_squares(
        residuals_function, initial_guess, bounds=(bounds_lower, bounds_upper),
        max_nfev=20000, verbose=2
    )

    k_labels = ['k_AE', 'k_AC', 'k_EI', 'k_CF', 'k_CB', 'k_FH', 'k_IH', 'k_HF', 'k_form']

    print("\nOptimized Arrhenius parameters (A_i and E_a):")
    for i, label in enumerate(k_labels[:-1]):
        A_i = result.x[2 * i]
        E_i = result.x[2 * i + 1]
        print(f"{label}: A = {A_i:.6e} L/(mol·s), E_a = {E_i:.2f} J/mol")
    print(f"k_form: {result.x[-1]:.6f} mol/(L·s)")

    # =========================
    # Jacobian-based 95% CIs
    # =========================
    try:
        J = result.jac                                  # (n_obs x n_params)
        resid = result.fun
        n_obs, n_params = J.shape
        dof = max(n_obs - n_params, 1)
        ssr_j = float(resid @ resid)
        s2 = ssr_j / dof                                # residual variance estimate

        JTJ = J.T @ J
        JTJ_inv = np.linalg.pinv(JTJ)                   # stable inverse
        cov_params = s2 * JTJ_inv                       # (p x p)
        se_params = np.sqrt(np.clip(np.diag(cov_params), 0.0, np.inf))
        z = 1.96

        print("\n95% Confidence Intervals for fitted parameters (optimization space):")
        for i in range(len(result.x)):
            theta = result.x[i]
            se = se_params[i]
            lo, hi = theta - z * se, theta + z * se
            if i < 16:
                # map back to label and type (A or E)
                k_idx = i // 2
                is_A = (i % 2 == 0)
                lbl = f"{k_labels[k_idx]}_{'A' if is_A else 'E'}"
            else:
                lbl = "k_form"
            print(f"  {lbl}: {theta:.6g}  (SE={se:.3g})  95% CI: [{lo:.6g}, {hi:.6g}]")

        # Explicit A and Ea CI printout per Arrhenius pair
        print("\n95% Confidence Intervals for A and Ea (by reaction):")
        for i in range(8):  # 8 Arrhenius reactions
            A_est = result.x[2*i]
            E_est = result.x[2*i + 1]
            se_A  = se_params[2*i]
            se_E  = se_params[2*i + 1]
            lo_A, hi_A = A_est - z * se_A, A_est + z * se_A
            lo_E, hi_E = E_est - z * se_E, E_est + z * se_E
            lbl = k_labels[i]
            print(f"{lbl}:")
            print(f"   A  = {A_est:.6e}  (SE={se_A:.3g})  95% CI: [{lo_A:.6e}, {hi_A:.6e}]")
            print(f"   Ea = {E_est:.2f} J/mol  (SE={se_E:.3g})  95% CI: [{lo_E:.2f}, {hi_E:.2f}]")

        # (Optional) bounds proximity note
        lb = np.array([0]*16 + [0.0])
        ub = np.array([1e12 if i%2==0 else 1e5 for i in range(16)] + [10.0])
        active_lower = np.isclose(result.x, lb, rtol=0, atol=1e-10*(1+np.abs(lb)))
        active_upper = np.isclose(result.x, ub, rtol=0, atol=1e-10*(1+np.abs(ub)))
        if np.any(active_lower | active_upper):
            hit = np.where(active_lower, "lower", np.where(active_upper, "upper", "")) 
            idxs = np.where(active_lower | active_upper)[0]
            print("\n[Note] Some parameters hit bounds; asymptotic CIs may be unreliable for:")
            for i in idxs:
                if i < 16:
                    k_idx = i // 2
                    is_A = (i % 2 == 0)
                    lbl = f"{k_labels[k_idx]}_{'A' if is_A else 'E'}"
                else:
                    lbl = "k_form"
                print(f"  {lbl} at {hit[i]} bound")

        # =========================
        # Diagnostics: conditioning & correlations
        # =========================
        try:
            w = np.linalg.eigvalsh(JTJ)
            condnum = (w.max() / w.min()) if w.min() > 0 else np.inf
            print(f"\n[Diag] cond(J^T J) ≈ {condnum:.3e}  (>>1e8 is very ill-conditioned)")

            D = np.sqrt(np.clip(np.diag(cov_params), 0, np.inf))
            with np.errstate(divide='ignore', invalid='ignore'):
                corr = cov_params / np.outer(D, D)

            # ---- Descriptive reaction names for diagnostics ----
            reaction_map = {
                'k_AE': "4-HBA→4-HBA Ester",
                'k_AC': "4-HBA→Mono-Alkylated",
                'k_EI': "4-HBA Ester→Mono-Alkylated Ester",
                'k_CF': "Mono-Alkylated→DIPBA",
                'k_CB': "Mono-Alkylated→Sulfonated Impurity",
                'k_FH': "DIPBA→DIPBA Ester",
                'k_IH': "Mono-Alkylated Ester→DIPBA Ester",
                'k_HF': "DIPBA Ester→DIPBA",
                'k_form': "Carbocation Formation"
            }

            # Build descriptive parameter names (A/Ea for first 8, then k_form)
            descriptive_names = []
            for i in range(16):
                rxn_key = k_labels[i//2]
                suffix = "A" if (i % 2 == 0) else "Ea"
                descriptive_names.append(f"{reaction_map[rxn_key]}_{suffix}")
            descriptive_names.append(reaction_map['k_form'])

            # Print top correlations with descriptive names
            pairs = []
            for i in range(len(result.x)):
                for j in range(i+1, len(result.x)):
                    c = corr[i, j]
                    if np.isfinite(c):
                        pairs.append((abs(c), c, i, j))
            pairs.sort(reverse=True)
            print("\n[Diag] Top |correlation| parameter pairs (descriptive):")
            for k in range(min(10, len(pairs))):
                ac, c, i_idx, j_idx = pairs[k]
                print(f"  {descriptive_names[i_idx]} ↔ {descriptive_names[j_idx]}: corr = {c:.3f}")
        except Exception as e:
            print(f"[Diag] Diagnostics failed: {e}")

        # =========================
        # 95% CIs for rate constants k(T)
        # =========================
        R = 8.314
        arr_params = result.x[:-1]  # 16 Arrhenius params (A,E pairs)
        k_labels_only = ['k_AE','k_AC','k_EI','k_CF','k_CB','k_FH','k_IH','k_HF']

        def k_ci_at_T(theta, cov, T):
            out = []
            for i in range(8):
                A = theta[2*i]; E = theta[2*i+1]
                k_hat = A * np.exp(-E/(R*T))
                dkdA = np.exp(-E/(R*T))
                dkdE = -A * np.exp(-E/(R*T)) * (1/(R*T))
                g = np.zeros(len(theta)+1)  # +1 because k_form is appended in result.x/cov
                g[2*i]   = dkdA
                g[2*i+1] = dkdE
                var_k = float(g @ cov @ g)
                se_k = np.sqrt(max(var_k, 0.0))
                out.append((k_hat, k_hat - z*se_k, k_hat + z*se_k))
            return out

        print("\n95% CIs for k(T) at each experiment's temperature:")
        for exp in experiments:
            T = temperatures[exp]
            vals = k_ci_at_T(arr_params, cov_params, T)
            print(f"  Exp {exp} at T={T:.2f} K:")
            for lbl, (khat, lo, hi) in zip(k_labels_only, vals):
                print(f"    {lbl}: {khat:.3e}  95% CI [{lo:.3e}, {hi:.3e}]")

        T_med = float(np.median(list(temperatures.values())))
        vals_med = k_ci_at_T(arr_params, cov_params, T_med)
        print(f"\n95% CIs for k(T) at median T={T_med:.2f} K:")
        for lbl, (khat, lo, hi) in zip(k_labels_only, vals_med):
            print(f"  {lbl}: {khat:.3e}  95% CI [{lo:.3e}, {hi:.3e}]")

        # k_form CI directly from covariance:
        se_kf = np.sqrt(max(cov_params[-1, -1], 0.0))
        kf_hat = result.x[-1]
        print(f"\nk_form: {kf_hat:.6f} mol/(L·s)  95% CI [{kf_hat - z*se_kf:.6f}, {kf_hat + z*se_kf:.6f}]")

        # =========================
        # Full covariance & correlation matrices + heatmaps (DESCRIPTIVE labels)
        # =========================
        # Build descriptive labels for the 17 parameters (16 Arrhenius + k_form)
        pretty_labels = []
        reaction_map_for_labels = {
            'k_AE': "4-HBA→4-HBA Ester",
            'k_AC': "4-HBA→Mono-Alkylated",
            'k_EI': "4-HBA Ester→Mono-Alkylated Ester",
            'k_CF': "Mono-Alkylated→DIPBA",
            'k_CB': "Mono-Alkylated→Sulfonated Impurity",
            'k_FH': "DIPBA→DIPBA Ester",
            'k_IH': "Mono-Alkylated Ester→DIPBA Ester",
            'k_HF': "DIPBA Ester→DIPBA",
            'k_form': "Carbocation Formation"
        }
        for i in range(8):
            pretty_labels.append(f"{reaction_map_for_labels[k_labels[i]]}_A")
            pretty_labels.append(f"{reaction_map_for_labels[k_labels[i]]}_Ea")
        pretty_labels.append(reaction_map_for_labels['k_form'])

        # Correlation matrix from covariance
        D = np.sqrt(np.clip(np.diag(cov_params), 0, np.inf))
        with np.errstate(divide='ignore', invalid='ignore'):
            corr_mat = cov_params / np.outer(D, D)

        # Save matrices (descriptive CSVs)
        cov_df = pd.DataFrame(cov_params, index=pretty_labels, columns=pretty_labels)
        corr_df = pd.DataFrame(corr_mat,   index=pretty_labels, columns=pretty_labels)
        cov_df.to_csv("covariance_matrix_descriptive.csv")
        corr_df.to_csv("correlation_matrix_descriptive.csv")
        print("\nSaved descriptive matrices: 'covariance_matrix_descriptive.csv' and 'correlation_matrix_descriptive.csv'.")

        # Heatmap: Covariance (descriptive)
        plt.figure(figsize=(10, 8))
        vmax_cov = np.nanmax(np.abs(cov_params))
        plt.imshow(cov_params, cmap='RdBu_r', interpolation='nearest', vmin=-vmax_cov, vmax=vmax_cov)
        plt.colorbar(label='Covariance')
        plt.xticks(ticks=np.arange(len(pretty_labels)), labels=pretty_labels, rotation=90)
        plt.yticks(ticks=np.arange(len(pretty_labels)), labels=pretty_labels)
        plt.title('Parameter Covariance Matrix (Descriptive Labels)')
        plt.tight_layout()
        plt.savefig("covariance_heatmap_descriptive.png", dpi=300)
        print("Saved 'covariance_heatmap_descriptive.png'.")
        plt.show()

        # Heatmap: Correlation (descriptive)
        plt.figure(figsize=(10, 8))
        plt.imshow(corr_mat, cmap='RdBu_r', interpolation='nearest', vmin=-1, vmax=1)
        plt.colorbar(label='Correlation')
        plt.xticks(ticks=np.arange(len(prety_labels := pretty_labels)), labels=pretty_labels, rotation=90)
        plt.yticks(ticks=np.arange(len(pretty_labels)), labels=pretty_labels)
        plt.title('Parameter Correlation Matrix (Descriptive Labels)')
        plt.tight_layout()
        plt.savefig("correlation_heatmap_descriptive.png", dpi=300)
        print("Saved 'correlation_heatmap_descriptive.png'.")
        plt.show()

    except Exception as e:
        print(f"\n[Warning] Could not compute Jacobian-based CIs or matrices: {e}")

    residuals = result.fun
    ssr = np.sum(residuals**2)
    n_total = len(residuals)
    rmse_overall = np.sqrt(ssr / n_total)
    ss_total = np.sum((residuals - np.mean(residuals))**2)
    r_squared_total = 1 - (ssr / ss_total)
    # For overall NRMSE, we can normalize by the overall range of residuals if desired.
    # Here, as an example, we compute the NRMSE for each compound separately.
    print(f"\nSum of Squared Residuals (SSR): {ssr}")
    print(f"Overall RMSE: {rmse_overall:.6f}")
    print(f"Overall R²: {r_squared_total:.3f}")

    for compound in compounds:
        actual_data = []
        fitted_data = []
        index = species_indices[compound]
        for time_data, init_conditions, exp, temperature in zip(
            combined_time_data, initial_conditions_list, experiments_list, temperature_list
        ):
            try:
                actual_conc = all_data[compound]['concentration'][exp]
                sol = solve_kinetics_wrapper_for_experiment(
                    time_data, result.x[:-1], init_conditions, temperature, result.x[-1]
                )
                model_conc = sol[index, :]
                actual_data.extend(actual_conc)
                fitted_data.extend(model_conc)
            except RuntimeError as e:
                print(f"Skipping experiment {exp} for compound {compound} due to ODE solver failure: {e}")
                continue
        actual_data = np.array(actual_data)
        fitted_data = np.array(fitted_data)
        resid = actual_data - fitted_data
        n_comp = len(actual_data)
        rmse_comp = np.sqrt(np.sum(resid**2) / n_comp)
        data_range = np.max(actual_data) - np.min(actual_data)
        nrmse_comp = rmse_comp / data_range if data_range != 0 else rmse_comp
        ssr_comp = np.sum(resid**2)
        ss_total_comp = np.sum((actual_data - np.mean(actual_data))**2)
        r2_comp = 1 - (ssr_comp / ss_total_comp)
        print(f"R² for {compound}: {r2_comp:.3f}, RMSE: {rmse_comp:.6f}, NRMSE: {nrmse_comp:.6f}")

    # Plot actual vs fitted for compound A as an example
    plt.figure(figsize=(10, 6))
    for exp in experiments:
        t_data = all_data['A']['time'][exp]
        actual_conc = all_data['A']['concentration'][exp]
        init_conditions = initial_conditions_list[experiments_list.index(exp)]
        try:
            sol = solve_kinetics_wrapper_for_experiment(
                t_data, result.x[:-1], init_conditions, temperature=temperatures[exp], k_form=result.x[-1]
            )
            model_conc = sol[species_indices['A'], :]
            plt.plot(t_data, actual_conc, 'o', label=f'Actual A - Exp {exp}')
            plt.plot(t_data, model_conc, '-', label=f'Fitted A - Exp {exp}')
        except RuntimeError as e:
            print(f"Skipping plotting for experiment {exp} due to ODE solver failure: {e}")
            continue
    plt.xlabel('Time (s)')
    plt.ylabel('Concentration of A (mol/L)')
    plt.title('Actual vs Fitted Concentration of A Over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
