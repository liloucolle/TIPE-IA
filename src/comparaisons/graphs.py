# graphes.py
import os
from datetime import datetime
import matplotlib.pyplot as plt

from comparaison import *
from content.echantillon_texte import (
    test_simple,
    texte_king_and_queen,
    texte_king_and_queen_2,
    texte_king_and_queen_long,   
    texte_science
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
    #("test_simple", test_simple),
    ("king_and_queen", texte_king_and_queen),
    ("science", texte_science),
    #("king_and_queen_long", texte_king_and_queen_long),  
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

"""
# =========================
# 2) Précision top-k
# =========================
def graphe_precision(k=TOPK):
    labels_models = ["Softmax (NumPy)", "NegSampling (NumPy)", "NegSampling (PyTorch)"]

    acc_soft = []
    acc_neg  = []
    acc_pt   = []

    for name, corpus in CORPORA:
        _, _, _, _, _, paires = build_dataset(corpus, window_size=params_numpy["window_size"])

        W_soft, _ = benchmark_softmax_numpy(corpus, params_numpy)
        W_neg,  _ = benchmark_neg_numpy(corpus, {**params_numpy, "K_neg": K_NEG})
        W_pt, _, paires_pt = benchmark_pytorch(corpus, mode="neg", epochs=params_numpy["n_epochs"])

        a_soft = topk_accuracy(W_soft, paires, k=k)
        a_neg  = topk_accuracy(W_neg,  paires, k=k)
        a_pt   = topk_accuracy(W_pt,   paires_pt, k=k)

        acc_soft.append(a_soft)
        acc_neg.append(a_neg)
        acc_pt.append(a_pt)

        print(f"[ACC@{k}] {name}: soft={a_soft:.4f} | neg_np={a_neg:.4f} | neg_pt={a_pt:.4f}")

    x = list(range(len(CORPORA)))
    width = 0.25

    fig = plt.figure()
    plt.bar([i - width for i in x], acc_soft, width=width, label=labels_models[0])
    plt.bar([i for i in x],         acc_neg,  width=width, label=labels_models[1])
    plt.bar([i + width for i in x], acc_pt,   width=width, label=labels_models[2])

    plt.xticks(x, [name for name, _ in CORPORA], rotation=0)
    plt.ylabel(f"Top-{k} accuracy")
    plt.ylim(0.0, 1.0)
    plt.title(f"Qualité sémantique (Top-{k}) — 3 implémentations, 4 corpus")
    plt.legend()

    save_figure(fig, f"precision_top{k}_4corpus")
    plt.show()

"""
# =========================
# 3) Lancement
# =========================
if __name__ == "__main__":
    graphe_temps()
    #graphe_precision(k=TOPK)
