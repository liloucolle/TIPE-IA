# graphes.py
import os
from datetime import datetime
import matplotlib.pyplot as plt

from comparaison import *
from content.echantillon_texte import (
   texte_diversifie,texte_redundant,texte_court,texte_structurel
)


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
    ("test_court", texte_court),
    ("Texte redondant", texte_redundant),
    ("Texte structurel", texte_structurel),
    ("Texte diversifié", texte_diversifie)
]


# =========================
# 1) Temps d'entraînement
# =========================
def graphe_temps():
    labels_models = ["Softmax (NumPy)", "NegSampling (NumPy)", "NegSampling (PyTorch)"]

    times_soft = []
    times_neg  = []
    times_pt   = []


    for name, corpus in CORPORA:
        _, t_soft = benchmark_softmax_numpy(corpus, params_numpy)
        _, t_neg  = benchmark_neg_numpy(corpus, {**params_numpy, "K_neg": K_NEG})

        benchmark_pytorch(corpus, params_torch)

        _, t_pt = benchmark_pytorch(corpus, params_torch)

        times_soft.append(t_soft)
        times_neg.append(t_neg)
        times_pt.append(t_pt)

        print(f"[TEMPS] {name}: soft={t_soft:.4f}s | neg_np={t_neg:.4f}s | neg_pt={t_pt:.4f}s")

    x = list(range(len(CORPORA)))
    width = 0.25

    fig = plt.figure()
    plt.bar([i - width for i in x], times_soft, width=width, label=labels_models[0])
    plt.bar([i for i in x],         times_neg,  width=width, label=labels_models[1])
    plt.bar([i + width for i in x], times_pt,   width=width, label=labels_models[2])

    plt.xticks(x, [name for name, _ in CORPORA], rotation=0)
    plt.ylabel("Temps (s)")
    plt.title("Temps d'entraînement Word2Vec (3 implémentations, 4 corpus)")
    plt.legend()

    save_figure(fig, "temps_word2vec_4corpus")
    plt.show()


# =========================
# 2) Convergence (loss proxy)
# =========================
def graphe_convergence():
    for name, corpus in CORPORA:
        res = test_convergence_proxy(
            corpus,
            params_numpy=params_numpy,
            params_torch=params_torch
        )

        fig = plt.figure()
        plt.plot(res["epochs"], res["softmax_numpy"], label="Softmax NumPy")
        plt.plot(res["epochs"], res["neg_numpy"], label="NegSampling NumPy")
        plt.plot(res["epochs"], res["pytorch"], label="NegSampling PyTorch")

        plt.xlabel("Epochs")
        plt.ylabel("Proxy loss (↓ mieux)")
        plt.title(f"Convergence — {name}")
        plt.legend()

        save_figure(fig, f"convergence_{name}")
        plt.show()



# =========================
# 3) Scalabilité (temps vs vocabulaire) — VERSION PROPRE
# =========================
def graphe_scalabilite():
    V_all = []
    t_soft_all = []
    t_neg_all = []
    t_pt_all = []

    corpus_labels = []

    for name, corpus in CORPORA:
        res = test_scalabilite_vocab(
            corpus,
            params_numpy=params_numpy,
            params_torch=params_torch
        )

        # On concatène les résultats
        V_all.extend(res["V"])
        t_soft_all.extend(res["t_soft"])
        t_neg_all.extend(res["t_neg"])
        t_pt_all.extend(res["t_pt"])
        corpus_labels.extend([name] * len(res["V"]))

    # Tri par taille de vocabulaire (important pour une courbe propre)
    order = np.argsort(V_all)
    V_all = np.array(V_all)[order]
    t_soft_all = np.array(t_soft_all)[order]
    t_neg_all = np.array(t_neg_all)[order]
    t_pt_all = np.array(t_pt_all)[order]

    fig = plt.figure()

    plt.plot(V_all, t_soft_all, "o-", label="Softmax NumPy")
    plt.plot(V_all, t_neg_all, "s-", label="NegSampling NumPy")
    plt.plot(V_all, t_pt_all, "^-", label="NegSampling PyTorch")

    plt.xlabel("Taille du vocabulaire |V|")
    plt.ylabel("Temps d'entraînement (s)")
    plt.title("Scalabilité des implémentations Word2Vec")
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_figure(fig, "scalabilite_globale")
    plt.show()


if __name__ == "__main__":
    graphe_temps()
    graphe_convergence()
    graphe_scalabilite()
