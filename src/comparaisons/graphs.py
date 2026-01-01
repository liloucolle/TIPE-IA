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
import csv



COLORS = {
    "softmax": "#195CB9",   
    "neg_numpy": "#a8123f", 
    "pt_softmax": "#59830A",   
    "pt_neg": "#917109"
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
params_torch_neg = dict(
    dimension=100,
    taux_apprentissage=0.05,
    window_size=2,
    n_epochs=10,
    mode="neg",          
    batch_size=128
)

params_torch_softmax = dict(
    dimension=100,
    taux_apprentissage=0.05,
    window_size=2,
    n_epochs=10,
    mode="softmax",          
    batch_size=128
)


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
    ("Texte 1000", texte_1000),
    ("Texte 10000", texte_10000)
]

# =========================
# 1) Scalabilité 
# =========================
def graphe_scalibility():
    labels_models = ["Softmax (NumPy)", "Softmax (Pytorch)", "NegSampling (NumPy)", "NegSampling (PyTorch)"]
    csv_path = "scalibility.csv"

    times_soft = []
    times_neg  = []
    times_pt_softmax   = []
    times_pt_neg = []

    benchmark_pytorch(texte_10, params_torch_neg)  # Cold start
    n_pairs_list = []
    taille_vocab_list = []
    for name, corpus in CORPORA:
        print(f"\n--------------------------")
        print(f"[SCALABILITY] Traitement du corpus: {name}")
        res = test_vitesse(corpus, params_numpy, params_torch_softmax, params_torch_neg, csv_path)

        t_soft = res["softmax_numpy"]["time"]
        t_neg  = res["neg_numpy"]["time"]
        t_pt_soft  = res["pytorch_softmax"]["time"]
        t_pt_neg   = res["pytorch_neg_sampling"]["time"]
        n_pairs = res["n_pairs"]
        taille_vocab = res["taille_vocab"]

        n_pairs_list.append(n_pairs)
        taille_vocab_list.append(taille_vocab)
        times_soft.append(t_soft)
        times_neg.append(t_neg)
        times_pt_softmax.append(t_pt_soft)
        times_pt_neg.append(t_pt_neg)

        print(f"[TEMPS] {name}: pairs={n_pairs} |pt_softmax = {t_pt_soft:.4f}s | neg_pt={t_pt_neg:.4f}s")

    # x axis: number of pairs; y axis: total time (log scale)
    x = n_pairs_list

    fig, ax = plt.subplots()
    ax.plot(x, times_soft, marker='o', linestyle='-', label=labels_models[0], color=COLORS["softmax"])
    ax.plot(x, times_neg,  marker='o', linestyle='-', label=labels_models[1], color=COLORS["neg_numpy"])
    ax.plot(x, times_pt_softmax,   marker='o', linestyle='-', label=labels_models[2], color=COLORS["pt_softmax"])
    ax.plot(x, times_pt_neg,   marker='o', linestyle='-', label=labels_models[3], color=COLORS["pt_neg"])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v}\n({vocab})" for v, vocab in zip(x, taille_vocab_list)], rotation=0)
    ax.set_yscale('log')
    ax.set_xlabel('Nombre de paires\n(taille vocabulaire)')
    ax.set_ylabel('Temps total (s) — échelle logarithmique')
    ax.set_title('Scalabilité: temps total sur le nombre de paires (log Y)')
    ax.legend()

    save_figure(fig, "temps_word2vec_4corpus")
    plt.show()

    # =========================
    # Résumé & export CSV pour Excel
    # =========================
    # Construire les en-têtes et lignes
    headers = [
        "corpus",
        "n_pairs",
        "vocab_size",
        "time_soft_numpy_s",
        "time_neg_numpy_s",
        "time_pt_softmax_s",
        "time_pt_neg_s",
    ]

    rows = []
    # Use the collected timing lists (do not re-enumerate CORPORA contents)
    for i in range(len(n_pairs_list)):
        name = CORPORA[i][0] if i < len(CORPORA) else f"corpus_{i}"
        rows.append([
            name,
            n_pairs_list[i],
            taille_vocab_list[i],
            times_soft[i],
            times_neg[i],
            times_pt_softmax[i],
            times_pt_neg[i],
        ])

    # Pretty print table to console
    col_widths = [max(len(str(r[j])) for r in ([headers] + rows)) + 2 for j in range(len(headers))]
    fmt = "".join(f"{{:<{w}}}" for w in col_widths)
    print("\n" + "="*60)
    print("Résumé des temps par corpus:")
    print("="*60)
    print(fmt.format(*headers))
    print("-" * sum(col_widths))
    for r in rows:
        # Format floats to 4 decimals for readability
        r_print = [r[0], r[1], r[2], f"{r[3]:.4f}", f"{r[4]:.4f}", f"{r[5]:.4f}", f"{r[6]:.4f}"]
        print(fmt.format(*r_print))

    # Save CSV for Excel import
    out_dir = "figures"
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(out_dir, f"times_word2vec_{stamp}.csv")
    with open(csv_path, "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for r in rows:
            writer.writerow(r)

    print(f"[OK] Temps sauvegardés en CSV : {csv_path}")


if __name__ == "__main__":
    graphe_scalibility()

