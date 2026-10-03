import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Caricamento dei dati generati dal C++
df = pd.read_csv("benchmark_varying_r.csv")

# Configurazione dello stile grafico "Paper-like"
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 8), sharex=True, sharey='row')

matrix_types = df["MatrixType"].unique()
targets = df["Target"].unique()

# Colori distinti per i metodi
palette = {"Hutchinson": "#d95f02", "Hutch++": "#1b9e77"}

for i, target in enumerate(targets):
    for j, m_type in enumerate(matrix_types):
        ax = axes[i, j]
        
        # Filtriamo il dataframe per il target e il tipo di matrice corrente
        subset = df[(df["Target"] == target) & (df["MatrixType"] == m_type)]
        
        # Calcoliamo mediana, 25° percentile e 75° percentile per ogni raggruppamento
        summary = subset.groupby(["r", "Method"])["RelError"].agg(
            median=lambda x: np.percentile(x, 50),
            q25=lambda x: np.percentile(x, 25),
            q75=lambda x: np.percentile(x, 75)
        ).reset_index()
        
        # Plot per ciascun metodo
        for method in ["Hutchinson", "Hutch++"]:
            m_data = summary[summary["Method"] == method]
            if m_data.empty:
                continue
                
            color = palette.get(method, "black")
            
            # Linea della mediana
            ax.plot(m_data["r"], m_data["median"], label=method, color=color, linewidth=1.5, marker='o', markersize=3)
            
            # Area ombreggiata (25° - 75° percentile)
            ax.fill_between(
                m_data["r"], 
                m_data["q25"], 
                m_data["q75"], 
                color=color, 
                alpha=0.2
            )
            
        # Impostazioni degli assi in scala logaritmica per l'errore
        ax.set_yscale('log')
        
        # Titoli dei pannelli (colonne in alto, righe a destra)
        if i == 0:
            ax.set_title(m_type, fontsize=11, fontweight='bold')
        if j == 2:
            ax.twinx().set_ylabel(target, rotation=270, labelpad=15, fontsize=10, fontweight='bold')
            # Nascondiamo i tick del secondo asse fittizio
            plt.gca().set_yticklabels([])
            plt.gca().set_yticks([])

        # Etichette degli assi
        if i == 1:
            ax.set_xlabel("Numero di Query (r)", fontsize=10)
        if j == 0:
            ax.set_ylabel("Errore Relativo (Log Scale)", fontsize=10)
            
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

# Gestione della legenda comune in alto
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=True, fontsize=11)

plt.suptitle("Confronto Convergenza: Hutchinson vs Hutch++ (Mediana e Intervallo Interquartile)", fontsize=14, fontweight='bold', y=1.06)
plt.tight_layout()

# Salvataggio dell'immagine ad alta risoluzione pronta per la tesi
plt.savefig("paper_style_convergence.png", dpi=300, bbox_inches='tight')
plt.show()
print("Grafico in stile paper generato e salvato come 'paper_style_convergence.png'!")