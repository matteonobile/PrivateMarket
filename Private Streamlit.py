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
@st.cache_data
def calculate_all_metrics(h_pe, h_pd, p_haircut, p_normal):
    shape = (len(steps), len(steps))
    mat_sharpe_w = np.zeros(shape)
    mat_risk_contrib = np.zeros(shape)
    mat_pdi = np.zeros(shape)
    
    for i, pct_pe in enumerate(steps):
        for j, pct_pd in enumerate(steps):
            w_pe = w_eq_tot * pct_pe
            w_pbeq = w_eq_tot - w_pe
            w_pd = w_bd_tot * pct_pd
            w_pbbd = w_bd_tot - w_pd
            w = np.array([w_pbeq, w_pe, w_pbbd, w_pd, w_alt, w_cash])
            
            port_ret = np.dot(w, mu)
            port_vol = np.sqrt(np.dot(w.T, np.dot(sigma, w)))
            
            # Sharpe Ponderata
            penalty = (w_pe * h_pe) + (w_pd * h_pd)
            sharpe_norm = (port_ret - risk_free) / port_vol
            sharpe_hair = ((port_ret - penalty) - risk_free) / port_vol
            mat_sharpe_w[i, j] = (sharpe_norm * p_normal) + (sharpe_hair * p_haircut)
            
            # Risk Contribution (Private)
            mcr = np.dot(sigma, w) / port_vol
            rc_abs = w * mcr
            mat_risk_contrib[i, j] = (rc_abs[1] + rc_abs[3]) / port_vol
            
            # Diversification Ratio
# 3. PDI (Morgan & Rudin - PCA Based)
            # A. Costruiamo la matrice di covarianza pesata: Elemento (i,j) = wi * wj * cov(i,j)
            w_diag = np.diag(w)
            weighted_cov_matrix = w_diag @ sigma @ w_diag
            
            # B. Calcolo Autovalori (Eigenvalues)
            # eigvalsh è ottimizzato per matrici simmetriche (come la covarianza)
            eigenvalues = np.linalg.eigvalsh(weighted_cov_matrix)
            
            # C. Pulizia (rimuoviamo eventuali valori negativi minuscoli dovuti a errori floating point)
            eigenvalues = np.maximum(eigenvalues, 0)
            
            # D. Calcolo PDI (Inverse Herfindahl of Eigenvalues)
            # Rappresenta il numero effettivo di fattori di rischio indipendenti
            sum_eig = np.sum(eigenvalues)
            if sum_eig > 0:
                p = eigenvalues / sum_eig # Normalizzazione (somma = 1)
                pdi = 1 / np.sum(p**2)    # Inverse sum of squares
            else:
                pdi = 1.0
                
            mat_pdi[i, j] = pdi
            
    return (pd.DataFrame(mat_sharpe_w, index=labels, columns=labels),
            pd.DataFrame(mat_risk_contrib, index=labels, columns=labels),
            pd.DataFrame(mat_pdi, index=labels, columns=labels))

df_sharpe, df_rc, df_div = calculate_all_metrics(haircut_pe, haircut_pd, prob_haircut, prob_normal)

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
         "Private Risk Contribution (%)", 
         "Diversification")
    )
    
    # Logica di selezione DataFrame e Formattazione
    if metric_choice == "Sharpe Ratio (weighted)":
        df_selected = df_sharpe
        cmap_selected = 'RdYlGn' # Verde = Bene
        fmt_selected = ".2f"     # 2 decimali per risparmiare spazio
        title_selected = "Weighted Sharpe Ratio"
        
    elif metric_choice == "Private Risk Contribution (%)":
        df_selected = df_rc
        cmap_selected = 'Reds'   # Rosso = Alto Rischio
        fmt_selected = ".0%"     # Percentuale senza decimali (es. 25%) per spazio
        title_selected = "% Risk Contribution from Private Assets"
        
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