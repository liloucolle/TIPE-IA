# comparaison_temporelle.py
import os
from datetime import datetime

from comparaison import *
from content.echantillon_texte import (
   texte_10,
   texte_100, 
   texte_1000,
   texte_10000,
)
import csv



# =========================
# Paramètres communs
# =========================
params_numpy = dict(
    dimension=100,
    taux_apprentissage=0.05,
    window_size=2,
    n_epochs=10,
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
# Textes (corpus) — 4 corpus ✅
# =========================
CORPORA = [
    ("Texte 10", texte_10),
    #("Texte 100", texte_100),
    #("Texte 1000", texte_1000),
    #("Texte 10000", texte_10000)
]

# =========================
# Génération CSV
# =========================
def create_benchmark_csv():
    """
    Génère un fichier CSV avec les temps de benchmark pour tous les corpus.
    """
    # Initialisation CSV
    out_dir = "figures"
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(out_dir, f"comparaison_temporelle_{stamp}.csv")

    headers = [
        "corpus",
        "n_pairs",
        "vocab_size",
        "time_soft_numpy_s",
        "time_neg_numpy_s",
        "time_pt_softmax_s",
        "time_pt_neg_s",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(headers)

        for name, corpus in CORPORA:
            print(f"\n--------------------------")
            print(f"[BENCHMARK] Corpus: {name}")
            res = test_vitesse(corpus, params_numpy, params_torch_softmax, params_torch_neg)

            t_pt_soft  = res["pytorch_softmax"]["time"]
            t_pt_neg   = res["pytorch_neg_sampling"]["time"]
            t_soft = res["softmax_numpy"]["time"]
            t_neg  = res["neg_numpy"]["time"]
            n_pairs = res["n_pairs"]
            taille_vocab = res["taille_vocab"]

            writer.writerow([name, n_pairs, taille_vocab, t_soft, t_neg, t_pt_soft, t_pt_neg])
            csv_file.flush()

            print(f"[TIMES] {name}:")
            print(f"  - Softmax NumPy:       {t_soft:.4f}s")
            print(f"  - Softmax PyTorch:     {t_pt_soft:.4f}s")
            print(f"  - NegSampling NumPy:   {t_neg:.4f}s")
            print(f"  - NegSampling PyTorch: {t_pt_neg:.4f}s")
            print(f"  - Pairs: {n_pairs} | Vocab: {taille_vocab}")

    print(f"\n[OK] CSV saved: {os.path.abspath(csv_path)}")



if __name__ == "__main__":
    create_benchmark_csv()
