# graphes.py
import matplotlib.pyplot as plt
from comparaison import *

from content.echantillon_texte import test_simple


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


# =========================
# 1) Temps d'entraînement
# =========================

def graphe_temps():
    W_soft, t_soft = benchmark_softmax_numpy(test_simple, params_numpy)
    W_neg,  t_neg  = benchmark_neg_numpy(test_simple, {**params_numpy, "K_neg": 5})
    _, t_pt, _ = benchmark_pytorch(test_simple, mode="neg", epochs=10)

    labels = ["Softmax (NumPy)", "NegSampling (NumPy)", "NegSampling (PyTorch)"]
    times = [t_soft, t_neg, t_pt]

    plt.bar(labels, times)
    plt.ylabel("Temps (s)")
    plt.title("Temps d'entraînement Word2Vec")
    plt.show()


# =========================
# 2) Précision top-k
# =========================

def graphe_precision():
    tokens, freq, vocab, stoi, itos, paires = build_dataset(test_simple)

    W_soft, _ = benchmark_softmax_numpy(test_simple, params_numpy)
    W_neg, _  = benchmark_neg_numpy(test_simple, {**params_numpy, "K_neg": 5})
    W_pt, _, paires_pt = benchmark_pytorch(test_simple, mode="neg", epochs=10)

    acc_soft = topk_accuracy(W_soft, paires, k=5)
    acc_neg  = topk_accuracy(W_neg,  paires, k=5)
    acc_pt   = topk_accuracy(W_pt,   paires_pt, k=5)

    labels = ["Softmax", "NegSampling", "PyTorch"]
    accs = [acc_soft, acc_neg, acc_pt]

    plt.bar(labels, accs)
    plt.ylabel("Top-5 accuracy")
    plt.title("Qualité sémantique (top-k)")
    plt.show()


# =========================
# 3) Lancement
# =========================

if __name__ == "__main__":
    graphe_temps()
    graphe_precision()
