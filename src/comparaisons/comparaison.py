# comparaisons.py
import csv
import time
import numpy as np

# ===== Imports de TES implémentations =====
from word2vec.word2vec import deep_learning as softmax_numpy
from word2vec_negativesampling.word2vec_negativesampling import deep_learning_negative_sampling
from word2vec_pytorch.word2vec_pytorch import (
    SkipGram, SkipGramSoftmaxDS, SkipGramNegSampleDS,
    train_softmax, train_neg
)

from content.echantillon_texte import texte_10

from text_utils.text_utils import tokenisation, frequence_mots, vocabulaire, creation_paires
import torch


# =========================
# Utils généraux
# =========================

def canon_texte(texte):
    """Assure texte = list[str] (corpus) pour être compatible partout."""
    if isinstance(texte, str):
        return [texte]
    return texte

def set_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)


# =========================
# Mesures de performance
# =========================

def benchmark_softmax_numpy(texte, params):
    set_seed()
    t0 = time.time()
    Wc, Wt= softmax_numpy(texte, **params)
    t = time.time() - t0
    return Wc, t


def benchmark_neg_numpy(texte, params):
    set_seed()
    t0 = time.time()
    Wc, Wt = deep_learning_negative_sampling(texte, **params)
    t = time.time() - t0

    return Wc, t


def benchmark_pytorch(texte, params):
    set_seed()
    t0 = time.time()
    from word2vec_pytorch.word2vec_pytorch import deep_learning_pytorch
    Wc = deep_learning_pytorch(texte, **params)
    t = time.time() - t0
    return Wc, t


def build_dataset(texte, window_size=2):
    """
    Prépare vocabulaire + paires (centre, contexte)
    pour les métriques de comparaison.
    """

    tokens = tokenisation(texte)
    freq = frequence_mots(tokens)
    vocab, stoi, itos = vocabulaire(freq)
    paires = creation_paires(tokens, stoi, taille_fenetre=window_size)

    return tokens, freq, vocab, stoi, itos, paires


# --------------------------
# Test  vitesse
# --------------------------
def test_vitesse(texte, params_numpy, params_torch_softmax, params_torch_neg):
    _, _, vocabulaire, _, _, paires = build_dataset(texte, window_size=params_numpy["window_size"])
    n_pairs = max(1, len(paires))
    taille_vocab = len(vocabulaire)
    n_epochs = params_numpy["n_epochs"]

    # Pour éviter l'effet de démarrage à froid
    benchmark_pytorch(texte_10, params_torch_neg)

    W_pt_soft,   t_pt_soft   = benchmark_pytorch(texte, params_torch_softmax)

    print("Finished pytorch softmax benchmark, time:", t_pt_soft)

    W_pt_neg,    t_pt_neg    = benchmark_pytorch(texte, params_torch_neg)

    print("Finished pytorch negative sampling benchmark, time:", t_pt_neg)

    W_neg,  t_neg  = benchmark_neg_numpy(texte, {**params_numpy, "K_neg": params_numpy.get("K_neg", 5)})
    W_soft, t_soft = benchmark_softmax_numpy(texte, params_numpy)


    # Bloc de sauvegarde CSV désactivé pour l'instant

    return {
        "n_pairs": n_pairs,
        "taille_vocab": taille_vocab,
        "n_epochs": n_epochs,
        "softmax_numpy": {"time": t_soft, "time_per_pair": t_soft, "emb": W_soft},
        "neg_numpy":     {"time": t_neg,  "time_per_pair": t_neg, "emb": W_neg},
        "pytorch_softmax": {"time": t_pt_soft,   "time_per_pair": t_pt_soft,   "emb": W_pt_soft},
        "pytorch_neg_sampling": {"time": t_pt_neg,    "time_per_pair": t_pt_neg,    "emb": W_pt_neg},
    }
