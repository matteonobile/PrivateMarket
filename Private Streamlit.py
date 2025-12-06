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

haircut_pe = st.sidebar.slider("Haircut Private Equity (%)", 0.0, 50.0, 20.0, step=1.0) / 100
haircut_pd = st.sidebar.slider("Haircut Private Debt (%)", 0.0, 50.0, 10.0, step=1.0) / 100

st.sidebar.markdown("---")
st.sidebar.info(f"Estimated Scenario:\n\n**{prob_normal:.1%}** no liquidation\n**{prob_haircut:.1%}** Liquidation")

# ---------------------------------------------------------
# 3. DATI FINANZIARI (BACKEND)
# ---------------------------------------------------------
# Simuliamo i dati (nel tuo caso reale caricheresti un Excel)
assets = ['EQUITIES','Private Equity','BONDS','Private Debt','ALTERNATIVES','Cash USD']

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path_ret = os.path.join(current_dir,"SAA Posterior FX UKIN Aggregation USD 20250901.xlsx")
file_path_cov = os.path.join(current_dir,"SAA Covariance FX UKIN Aggregation USD 20250901.xlsx")

expected_returns = pd.read_excel(file_path_ret,sheet_name="Sheet1")
expected_returns.set_index('index',drop=True,inplace=True)
expected_returns = expected_returns.iloc[:,0]
expected_returns = expected_returns.loc[assets]
mu = expected_returns

cov_matrix = pd.read_excel(file_path_cov,sheet_name="Sheet1")
cov_matrix.set_index(cov_matrix.columns[0],inplace=True,drop=True)
cov_matrix = cov_matrix.loc[assets,assets]
cov_matrix.loc['Cash USD','Cash USD'] = 0.00001
sigma = cov_matrix
# sigma = D @ corr @ D

# Pesi Macro (Fissi)
w_eq_tot = 0.50
w_bd_tot = 0.35
w_alt = 0.10
w_cash = 0.05
risk_free = 0.0

# Steps per la simulazione (Grid 10x10 per velocità)
steps = np.linspace(0, 1, 11) 

# ---------------------------------------------------------
# 4. MOTORE DI CALCOLO
# ---------------------------------------------------------

# Cache per evitare di ricalcolare se non cambiano i parametri
@st.cache_data
def calculate_matrices(h_pe, h_pd, p_haircut, p_normal):
    
    # Matrici vuote
    idx_cols = [f"{int(x*100)}%" for x in steps]
    mat_sharpe_norm = pd.DataFrame(index=idx_cols, columns=idx_cols)
    mat_sharpe_hair = pd.DataFrame(index=idx_cols, columns=idx_cols)
    
    w_benchmark = np.array([w_eq_tot, 0, w_bd_tot, 0, w_alt, w_cash])

    for i, pct_pe in enumerate(steps):
        for j, pct_pd in enumerate(steps):
            
            # Pesi
            w_pe = w_eq_tot * pct_pe
            w_pbeq = w_eq_tot - w_pe
            w_pd = w_bd_tot * pct_pd
            w_pbbd = w_bd_tot - w_pd
            
            w = np.array([w_pbeq, w_pe, w_pbbd, w_pd, w_alt, w_cash])
            
            # 1. Metriche Standard (Scenario Normale)
            ret_norm = np.dot(w, mu)
            risk = np.sqrt(np.dot(w.T, np.dot(sigma, w)))
            sharpe_norm = (ret_norm - risk_free) / risk
            
            # 2. Metriche Haircut (Scenario Stress)
            # Sottraiamo la penalità al rendimento atteso
            penalty = (w_pe * h_pe) + (w_pd * h_pd)
            ret_hair = ret_norm - penalty
            sharpe_hair = (ret_hair - risk_free) / risk
            
            # Salvataggio
            r_label = idx_cols[i]
            c_label = idx_cols[j]
            mat_sharpe_norm.loc[r_label, c_label] = sharpe_norm
            mat_sharpe_hair.loc[r_label, c_label] = sharpe_hair
            
    # Conversione a float per calcoli successivi
    mat_sharpe_norm = mat_sharpe_norm.astype(float)
    mat_sharpe_hair = mat_sharpe_hair.astype(float)
    
    # 3. Calcolo Weighted Average Sharpe
    mat_final = (mat_sharpe_norm * p_normal) + (mat_sharpe_hair * p_haircut)
    
    return mat_final, mat_sharpe_norm, mat_sharpe_hair

# Esecuzione Calcolo
final_matrix, s_norm, s_hair = calculate_matrices(haircut_pe, haircut_pd, prob_haircut, prob_normal)

# ---------------------------------------------------------
# 5. VISUALIZZAZIONE PRINCIPALE
# ---------------------------------------------------------

col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("Weighted Sharpe Ratio Heatmap")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(final_matrix, 
                annot=True, 
                fmt=".3f", 
                cmap='RdYlGn', 
                ax=ax,
                cbar_kws={'label': 'Weighted Sharpe Ratio'})
    
    ax.set_ylabel('% Private Equity (on Tot Equity)')
    ax.set_xlabel('% Private Debt (on Tot Bond)')
    st.pyplot(fig)

with col2:
    st.subheader("Best situation")
    
    # Trova il massimo
    max_val = final_matrix.max().max()
    max_idx = final_matrix.stack().idxmax()
    
    st.metric(label="Highest weighted sharpe ratio", value=f"{max_val:.4f}")
    st.write(f"**Optimal Allocation:**")
    st.write(f"PE: **{max_idx[0]}**")
    st.write(f"PD: **{max_idx[1]}**")
    
    st.markdown("---")
    
    # Confronto con Normalità Pura (per vedere l'impatto)
    val_norm_at_max = s_norm.loc[max_idx[0], max_idx[1]]
    diff = val_norm_at_max - max_val
    
    st.write("Haircut Impact:")
    st.write(f"Sharps drop by **{diff:.4f}** vs no liquidation")

# ---------------------------------------------------------
# 6. DETTAGLI ESPANDIBILI
# ---------------------------------------------------------
with st.expander("Underlying sharpe ratios"):
    c1, c2 = st.columns(2)
    with c1:
        st.write("**No liquidation scenario (100% prob)**")
        st.dataframe(s_norm.style.background_gradient(cmap='RdYlGn', axis=None).format("{:.3f}"))
    with c2:
        st.write("**Liquidation scenario (100% prob)**")
        st.dataframe(s_hair.style.background_gradient(cmap='RdYlGn', axis=None).format("{:.3f}"))