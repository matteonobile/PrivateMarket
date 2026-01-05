#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  6 08:39:11 2025

@author: matteo
"""

import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


# ---------------------------------------------------------
# 4. MONTE CARLO ENGINE (Pre-computation)
# ---------------------------------------------------------
@st.cache_data
def generate_asset_paths(mu, sigma, n_sims=500, horizon=52):
    """
    Generates random paths for all assets once.
    Horizon = 52 (weekly steps for 1 year) is usually enough for a dashboard 
    and faster than 252 daily steps.
    """
    n_assets = len(mu)
    dt = 1/horizon
    
    # Cholesky Decomposition
    L = np.linalg.cholesky(sigma)
    
    # Generate random shocks: (Horizon, n_sims, n_assets)
    # We use standard normal random variables
    random_shocks = np.random.normal(0, 1, (horizon, n_sims, n_assets))
    
    # Correlate shocks: shock_t * L.T
    # We broadcast the multiplication to get correlated returns
    correlated_shocks = np.einsum('tsa,ba->tsb', random_shocks, L)
    
    # Drift + Diffusion (Geometric Brownian Motion approx for returns)
    # return = (mu - 0.5*sigma^2)*dt + sigma*dW
    # For simple portfolio sim, arithmetic return approx is often sufficient: mu*dt + shock*sqrt(dt)
    
    asset_returns = (mu.values * dt) + (correlated_shocks * np.sqrt(dt))
    
    return asset_returns # Shape: (Horizon, Sims, Assets)

def calculate_mdd_scenarios(asset_returns, w, w_pe_idx, w_pd_idx, h_pe, h_pd):
    """
    Calculates Normal MDD and Haircut MDD for a specific set of weights.
    Vectorized over 'n_sims' columns.
    """
    # 1. Create Portfolio Returns Trace: (Horizon, Sims)
    # Dot product: (Horizon, Sims, Assets) @ (Assets) -> (Horizon, Sims)
    port_returns = asset_returns @ w 
    
    # 2. Cumulative Wealth Path (Normal)
    wealth_path = np.cumprod(1 + port_returns, axis=0)
    wealth_path = np.vstack([np.ones((1, wealth_path.shape[1])), wealth_path]) # Add starting 1.0
    
    # 3. Normal Max Drawdown
    running_max = np.maximum.accumulate(wealth_path, axis=0)
    drawdowns = (wealth_path - running_max) / running_max
    mdd_normal = drawdowns.min(axis=0).mean() # Average of worst drawdowns across sims
    
    # 4. Haircut MDD Logic
    # "Apply haircut to PE/PD from the lowest point and leave series flat from that point"
    
    # Find the index of the minimum value for each simulation (the "Lowest Point")
    # argmin returns the index (time step) where the minimum occurs
    min_indices = np.argmin(wealth_path, axis=0)
    
    # Create a mask for "after the crash"
    n_steps, n_sims = wealth_path.shape
    rows = np.arange(n_steps)[:, None]
    
    # We need to reconstruct the path with the shock
    # This is complex to fully vectorize without heavy memory, so we iterate sims or use advanced masking.
    # Given n_sims=500, a fast numba loop or list comprehension is best.
    
    final_mdds_haircut = []
    
    # Pre-calculate private portion weights
    # Assuming rebalancing isn't happening every step, the weight drifts. 
    # For approximation: We apply the haircut to the *proportion* of wealth held in PE/PD.
    total_priv_w = w[w_pe_idx] + w[w_pd_idx]
    
    # Optimization: We only loop through simulations, not time steps
    for sim_i in range(n_sims):
        idx_crash = min_indices[sim_i]
        path = wealth_path[:, sim_i].copy()
        
        # If the crash happens at the very end, the flatline doesn't matter
        if idx_crash < n_steps - 1:
            # 1. Apply Haircut at the crash point
            # The value at idx_crash drops by: (Value * Priv_Weight * Haircut_Avg)
            # We approximate the private weight exposure as constant 'total_priv_w'
            
            # Weighted average haircut
            if total_priv_w > 0:
                avg_haircut = (w[w_pe_idx]*h_pe + w[w_pd_idx]*h_pd) / total_priv_w
                shock_val = path[idx_crash] * total_priv_w * avg_haircut
                path[idx_crash] -= shock_val
            
            # 2. Flatline subsequent returns? 
            # User said: "leave the time series flat from that point on"
            # This means Value[t] = Value[crash_point] for all t > crash_point
            path[idx_crash+1:] = path[idx_crash]
            
        # Recalculate MDD on this new "Haircut Path"
        r_max = np.maximum.accumulate(path)
        dd = (path - r_max) / r_max
        final_mdds_haircut.append(dd.min())
        
    mdd_haircut = np.mean(final_mdds_haircut)
    
    return mdd_normal, mdd_haircut


# # Stampa di debug (rimuovila quando hai risolto)
# st.error("DEBUG MODE ATTIVO")
# st.write(f"Cartella di lavoro corrente: {os.getcwd()}")
# st.write("File trovati in questa cartella:")
# st.code(os.listdir(os.getcwd()))

# ---------------------------------------------------------
# 1. SETUP PAGINA E TITOLI
# ---------------------------------------------------------
st.set_page_config(page_title="Portfolio Optimizer", layout="wide")

st.title("📊 Portfolio Simulation: Private Assets & Liquidity Risk")
st.markdown("""
Providing the impact of Private Assets (PE e PD) 
considering Haircut impact and probability
""")

# ---------------------------------------------------------
# 2. SIDEBAR - PARAMETRI UTENTE
# ---------------------------------------------------------
st.sidebar.header("1. Probabilities")

# Probabilità
prob_haircut_pct = st.sidebar.slider(
    "Probability of 'Haircut' (%)", 
    min_value=0.0, max_value=100.0, value=12.5, step=0.5
)
prob_haircut = prob_haircut_pct / 100.0
prob_normal = 1.0 - prob_haircut

st.sidebar.markdown("---")
st.sidebar.header("2. Impact of Haircut")
st.sidebar.markdown("*Discount applied to liquidate private position*")

haircut_pe = st.sidebar.slider("Haircut Private Equity (%)", 0.0, 50.0, 30.0, step=1.0) / 100
haircut_pd = st.sidebar.slider("Haircut Private Debt (%)", 0.0, 50.0, 10.0, step=1.0) / 100

st.sidebar.markdown("---")
st.sidebar.info(f"Estimated Scenario:\n\n**{prob_normal:.1%}** no liquidation\n**{prob_haircut:.1%}** Liquidation")

# ---------------------------------------------------------
# 3. DATI FINANZIARI (BACKEND)
# ---------------------------------------------------------
# Simuliamo i dati (nel tuo caso reale caricheresti un Excel)
assets = ['EQUITIES','Private Equity','BONDS','Private Debt','ALTERNATIVES','Cash USD']

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path_ret = os.path.join(current_dir,"Posterior.xlsx")
file_path_cov = os.path.join(current_dir,"Covariance.xlsx")

expected_returns = pd.read_excel(file_path_ret,sheet_name="Sheet1")
expected_returns.set_index('index',drop=True,inplace=True)
expected_returns = expected_returns.iloc[:,0]
expected_returns = expected_returns.loc[assets]
mu = expected_returns

cov_matrix = pd.read_excel(file_path_cov,sheet_name="Sheet1")
cov_matrix.set_index(cov_matrix.columns[0],inplace=True,drop=True)
cov_matrix = cov_matrix.loc[assets,assets].astype(float)
cov_matrix.loc['Cash USD','Cash USD'] = 0.00001
sigma = cov_matrix
# sigma = D @ corr @ D

# Pesi Macro (Fissi)
w_eq_tot = 0.50
w_bd_tot = 0.35
w_alt = 0.10
w_cash = 0.05
risk_free = 0.0

steps = np.linspace(0, 1, 21) 
labels = [f"{int(x*100)}%" for x in steps]

# --- Funzione Calcolo (Con Cache) ---
# --- Generate Paths ONCE (Global context or inside function with cache) ---
# n_sims=200 is a good balance for speed vs accuracy in a heatmap
sim_returns = generate_asset_paths(mu, sigma, n_sims=200, horizon=52)

@st.cache_data
def calculate_all_metrics(h_pe, h_pd, p_haircut, p_normal):
    shape = (len(steps), len(steps))
    
    # Existing Matrices
    mat_sharpe_w = np.zeros(shape)
    mat_risk_contrib = np.zeros(shape)
    mat_te_contrib = np.zeros(shape)
    mat_info_w = np.zeros(shape)
    mat_pdi = np.zeros(shape)
    
    # NEW Matrix
    mat_mdd_w = np.zeros(shape) # Weighted MDD
    
    # Asset indices for MDD (PE is index 1, PD is index 3 in your 'assets' list)
    idx_pe = 1
    idx_pd = 3
    
    for i, pct_pe in enumerate(steps):
        for j, pct_pd in enumerate(steps):
            
            # ... [Your existing weight calc code] ...
            w_pe = w_eq_tot * pct_pe
            w_pbeq = w_eq_tot - w_pe
            w_pd = w_bd_tot * pct_pd
            w_pbbd = w_bd_tot - w_pd
            w_bmk = np.array([w_eq_tot,0,w_bd_tot, 0,w_alt,w_cash])
            w = np.array([w_pbeq, w_pe, w_pbbd, w_pd, w_alt, w_cash])
            w_act = w - w_bmk

            # ... [Your existing return/vol calc code] ...
            port_ret = np.dot(w, mu)
            bmk_ret = np.dot(w_bmk,mu)
            act_ret = port_ret - bmk_ret
            penalty = (w_pe * h_pe) + (w_pd * h_pd)

            port_vol = np.sqrt(np.dot(w.T, np.dot(sigma, w)))
            port_te = np.sqrt(np.dot(w_act.T, np.dot(sigma,w_act)))
            
            # ... [Your Sharpe/Info/RiskContrib/PDI code] ...
            # (Copy your existing logic here)
            sharpe_norm = (port_ret - risk_free) / port_vol
            sharpe_hair = ((port_ret - penalty) - risk_free) / port_vol
            mat_sharpe_w[i, j] = (sharpe_norm * p_normal) + (sharpe_hair * p_haircut)
            
            # --- NEW MDD CALCULATION ---
            # We pass the pre-calculated 'sim_returns' to avoid generating them 441 times
            mdd_norm_val, mdd_hair_val = calculate_mdd_scenarios(
                sim_returns, w, idx_pe, idx_pd, h_pe, h_pd
            )
            
            # Weighted MDD
            mat_mdd_w[i, j] = (mdd_norm_val * p_normal) + (mdd_hair_val * p_haircut)
            
            # (Fill other existing matrices...)
            # Information Ratio
            act_ret_hair = port_ret - penalty - bmk_ret
            info_norm = act_ret / port_te
            info_hair = act_ret_hair / port_te
            mat_info_w[i,j] = (info_norm * p_normal) + (info_hair * p_haircut)
            
            # RC
            mcr = np.dot(sigma, w) / port_vol
            rc_abs = w * mcr
            mat_risk_contrib[i, j] = (rc_abs[1] + rc_abs[3]) / port_vol
            
            # TE
            if port_te > 0:
                mrc = np.dot(sigma,w_act) / port_te
                tc_abs = w_act * mrc
                mat_te_contrib[i,j] = (tc_abs[1] + tc_abs[3]) / port_te
            
            # PDI
            w_diag = np.diag(w)
            weighted_cov_matrix = w_diag @ sigma @ w_diag
            eigenvalues = np.linalg.eigvalsh(weighted_cov_matrix)
            eigenvalues = np.maximum(eigenvalues, 0)
            sum_eig = np.sum(eigenvalues)
            if sum_eig > 0:
                p = eigenvalues / sum_eig 
                pdi = 1 / np.sum(p**2)    
            else:
                pdi = 1.0
            mat_pdi[i, j] = pdi

    return (pd.DataFrame(mat_sharpe_w, index=labels, columns=labels),
            pd.DataFrame(mat_info_w, index=labels, columns = labels),
            pd.DataFrame(mat_risk_contrib, index=labels, columns=labels),
            pd.DataFrame(mat_te_contrib, index = labels, columns=labels),
            pd.DataFrame(mat_pdi, index=labels, columns=labels),
            pd.DataFrame(mat_mdd_w, index=labels, columns=labels)) # <--- Return NEW DF



df_sharpe, df_info, df_rc, df_te, df_div,df_mdd = calculate_all_metrics(haircut_pe, haircut_pd, prob_haircut, prob_normal)

# ---------------------------------------------------------
# VISUALIZZAZIONE SEABORN + DOWNLOAD
# ---------------------------------------------------------
st.divider()
col_ctrl, col_plot = st.columns([1, 3])

with col_ctrl:
    st.subheader("Impostazioni")
    metric_choice = st.radio(
        "Metric to visualize:",
        ("Sharpe Ratio (weighted)", 
         "Information Ratio (weighted)", 
         "Private Risk Contribution (%)", 
         "Private TE Contribution (%)",
         "Diversification",
         "Max Drawdown (weighted)" # <--- NEW OPTION
         )
    )
    
    # Logica di selezione DataFrame e Formattazione
    if metric_choice == "Sharpe Ratio (weighted)":
        df_selected = df_sharpe
        cmap_selected = 'RdYlGn' # Verde = Bene
        fmt_selected = ".2f"     # 2 decimali per risparmiare spazio
        title_selected = "Weighted Sharpe Ratio"

    elif metric_choice == "Information Ratio (weighted)":
        df_selected = df_info
        cmap_selected = 'RdYlGn' # Verde = Bene
        fmt_selected = ".2f"     # 2 decimali per risparmiare spazio
        title_selected = "Weighted Information Ratio"

    elif metric_choice == "Private Risk Contribution (%)":
        df_selected = df_rc
        cmap_selected = 'Reds'   # Rosso = Alto Rischio
        fmt_selected = ".0%"     # Percentuale senza decimali (es. 25%) per spazio
        title_selected = "% Risk Contribution from Private Assets"

    elif metric_choice == "Private TE Contribution (%)":
        df_selected = df_te
        cmap_selected = 'Reds'   # Rosso = Alto Rischio
        fmt_selected = ".0%"     # Percentuale senza decimali (es. 25%) per spazio
        title_selected = "% TE Contribution from Private Assets"

    elif metric_choice == "Max Drawdown (weighted)":
        df_selected = df_mdd
        cmap_selected = 'Reds_r' # Inverted Red (Lower drawdown is better, usually negative numbers)
        # MDD is usually negative (e.g., -0.20), so closer to 0 is better (green/lighter)
        # If your calculation returns positive percentage for drawdown (e.g. 0.20), use 'Reds'
        fmt_selected = ".1%"
        title_selected = "Weighted Max Drawdown"    

    
    else:
        df_selected = df_div
        cmap_selected = 'viridis'
        fmt_selected = ".2f"
        title_selected = "Diversification"

    st.markdown("---")
    st.info("💡 **Tip:** use bottom button to download data.")

with col_plot:
    st.subheader(title_selected)
    
    # Setup Figura Matplotlib
    # Aumentiamo la dimensione (14x12) per gestire la griglia 21x21
    fig, ax = plt.subplots(figsize=(14, 12)) 
    
    # Heatmap Seaborn
    sns.heatmap(df_selected, 
                annot=True, 
                fmt=fmt_selected, 
                cmap=cmap_selected, 
                linewidths=.5,
                ax=ax,
                annot_kws={"size": 7}, # <--- FONT PICCOLO (7pt)
                cbar_kws={'label': metric_choice})
    
    # Pulizia Assi
    ax.set_ylabel('% Private Equity (on Tot Equity)', fontsize=11)
    ax.set_xlabel('% Private Debt (on Tot Bond)', fontsize=11)
    plt.yticks(rotation=0, fontsize=9)
    plt.xticks(rotation=45, fontsize=9)
    
    st.pyplot(fig)
    
    # ---------------------------------------------------------
    # DOWNLOAD BUTTON (CSV)
    # ---------------------------------------------------------
    # Convertiamo il dataframe selezionato in CSV
    csv_buffer = df_selected.to_csv().encode('utf-8')
    
    st.download_button(
        label=f"📥 Download data {metric_choice} (CSV)",
        data=csv_buffer,
        file_name=f"matrix_{metric_choice.replace(' ', '_')}.csv",
        mime='text/csv',
    )