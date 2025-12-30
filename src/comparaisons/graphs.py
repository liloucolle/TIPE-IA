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
    "softmax": "#1f3a5f",   # bleu marine profond
    "neg_numpy": "#7b1e3a", # rouge bordeaux
    "neg_pt": "#6b8e23",    # vert kaki (olive)
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


    for name, corpus in CORPORA:

        res = test_vitesse_par_paire(corpus, params_numpy, params_torch)

        t_soft_pp = res["softmax_numpy"]["time_per_pair"]
        t_neg_pp  = res["neg_numpy"]["time_per_pair"]
        t_pt_pp   = res["pytorch"]["time_per_pair"]
        n_pairs  = res["n_pairs"]
        taille_vocab = res["taille_vocab"]

        times_soft.append(t_soft_pp)
        times_neg.append(t_neg_pp)
        times_pt.append(t_pt_pp)

        print(f"[TEMPS/PAIRE] {name}: soft={t_soft_pp:.6e}s | neg_np={t_neg_pp:.6e}s | neg_pt={t_pt_pp:.6e}s")

    x = list(range(len(CORPORA)))
    width = 0.25

    fig, ax = plt.subplots()
    ax.bar([i - width for i in x], times_soft, width=width, label=labels_models[0], color=COLORS["soft"])
    ax.bar([i for i in x],         times_neg,  width=width, label=labels_models[1], color=COLORS["neg_np"])
    ax.bar([i + width for i in x], times_pt,   width=width, label=labels_models[2], color=COLORS["neg_pt"])

    ax.set_xticks(x)
    ax.set_xticklabels([f"{n_pairs}\n({taille_vocab})" for name, _ in CORPORA], rotation=0)
    ax.set_yscale('log')
    ax.set_ylabel("Temps par paire (s) — échelle logarithmique")
    ax.set_title("Scalabilité: temps par paire (log) — 3 implémentations, 4 corpus")
    ax.legend()

    save_figure(fig, "temps_word2vec_4corpus")
    plt.show()




if __name__ == "__main__":
    graphe_scalibility()
    #graphe_precision()
    #graphe_scalabilite()
