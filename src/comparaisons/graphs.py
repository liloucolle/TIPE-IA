# graphes.py
import os
from datetime import datetime
import matplotlib.pyplot as plt

from comparaison import *
from content.echantillon_texte import (
   texte_10,
   texte_100, 
   texte_1000,
   texte_10000,
   texte_100000
)


COLORS = {
    "softmax": "#385071",   
    "neg_numpy": "#893e55", 
    "neg_pt": "#6c8933",    
}

# =========================
# Paramètres communs
# =========================
params_numpy = dict(
    dimension=100,
    taux_apprentissage=0.05,
    window_size=2,
    n_epochs=10,
    patience=3,
    tolerance=1e-4,
    decay=0.95
)



# Params PyTorch (alignés sur la signature demandée)
params_torch = dict(
    dimension=100,
    taux_apprentissage=0.01,
    window_size=2,
    n_epochs=10,
    mode="neg",          # ✅ on compare NS pytorch avec NS numpy
    batch_size=128
)


K_NEG = 5
TOPK = 5





# =========================
# Utilitaires sauvegarde
# =========================
def save_figure(fig, name, out_dir="figures", dpi=200):
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"{name}_{stamp}.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"[OK] Figure sauvegardée : {path}")
    return path



# =========================
# Textes (corpus) — 4 corpus ✅
# =========================
CORPORA = [
    ("Texte 10", texte_10),
    ("Texte 100", texte_100),
    #("Texte 1000", texte_1000),
    #("Texte 10000", texte_10000)
  
]


# =========================
# 1) Scalabilité 
# =========================
def graphe_scalibility():
    labels_models = ["Softmax (NumPy)", "NegSampling (NumPy)", "NegSampling (PyTorch)"]

    times_soft = []
    times_neg  = []
    times_pt   = []


    n_pairs_list = []
    for name, corpus in CORPORA:
        res = test_vitesse(corpus, params_numpy, params_torch)

        t_soft = res["softmax_numpy"]["time"]
        t_neg  = res["neg_numpy"]["time"]
        t_pt   = res["pytorch"]["time"]
        n_pairs = res["n_pairs"]
        taille_vocab = res["taille_vocab"]

        n_pairs_list.append(n_pairs)
        times_soft.append(t_soft)
        times_neg.append(t_neg)
        times_pt.append(t_pt)

        print(f"[TEMPS] {name}: pairs={n_pairs} | soft={t_soft:.4f}s | neg_np={t_neg:.4f}s | neg_pt={t_pt:.4f}s")

    # x axis: number of pairs; y axis: total time (log scale)
    x = n_pairs_list

    fig, ax = plt.subplots()
    ax.plot(x, times_soft, marker='o', linestyle='-', label=labels_models[0], color=COLORS["softmax"])
    ax.plot(x, times_neg,  marker='o', linestyle='-', label=labels_models[1], color=COLORS["neg_numpy"])
    ax.plot(x, times_pt,   marker='o', linestyle='-', label=labels_models[2], color=COLORS["neg_pt"])

    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in x], rotation=0)
    ax.set_yscale('log')
    ax.set_xlabel('Nombre de paires')
    ax.set_ylabel('Temps total (s) — échelle logarithmique')
    ax.set_title('Scalabilité: temps total vs nombre de paires (log Y)')
    ax.legend()

    save_figure(fig, "temps_word2vec_4corpus")
    plt.show()




if __name__ == "__main__":
    graphe_scalibility()
    #graphe_precision()
    #graphe_scalabilite()
