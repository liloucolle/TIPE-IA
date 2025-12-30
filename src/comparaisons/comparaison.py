# comparaisons.py
import time
import numpy as np

# ===== Imports de TES implémentations =====
from word2vec.matrice import deep_learning as softmax_numpy
from word2vec_negativesampling.word2vec_negativesampling import deep_learning_negative_sampling
from word2vec_pytorch.word2vec_pytorch import (
    SkipGram, SkipGramSoftmaxDS, SkipGramNegSampleDS,
    train_softmax, train_neg
)

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

def normalize_rows(M):
    n = np.linalg.norm(M, axis=1, keepdims=True) + 1e-12
    return M / n

def cosine(u, v):
    return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))

def topk_accuracy(emb, paires, k=5, max_pairs=None):
    """
    Pour chaque (centre, contexte), est-ce que le contexte est dans les k plus proches voisins du centre ?
    """
    if len(paires) == 0:
        return 0.0
    if max_pairs is not None:
        paires = paires[:max_pairs]

    E = normalize_rows(emb)
    correct = 0
    for c, pos in paires:
        sims = E @ E[c]
        topk_idx = np.argsort(-sims)[:k+1]  # +1 car le centre lui-même peut être top1
        if pos in topk_idx:
            correct += 1
    return correct / len(paires)

# =========================
# Benchmarks 
# =========================

def benchmark_softmax_numpy(texte, params):
    set_seed()
    t0 = time.time()
    Wc, Wt = softmax_numpy(texte, **params)
    t = time.time() - t0
    #print(str(t) + ":total time benchamark softmax numpy implementation")
    return Wc, t


def benchmark_neg_numpy(texte, params):
    set_seed()
    t0 = time.time()
    Wc, Wt = deep_learning_negative_sampling(texte, **params)
    t = time.time() - t0
    #print(str(t) + ":total time benchamark negative sampling numpy implementation")

    return Wc, t


def benchmark_pytorch(texte, params):
    set_seed()
    t0 = time.time()
    from word2vec_pytorch.word2vec_pytorch import deep_learning_pytorch
    Wc = deep_learning_pytorch(texte, **params)
    t = time.time() - t0
    #print(str(t) + ":total time benchmark pytorch implementation")
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

    #W_soft, t_soft = benchmark_softmax_numpy(texte, params_numpy)
    #W_neg,  t_neg  = benchmark_neg_numpy(texte, {**params_numpy, "K_neg": params_numpy.get("K_neg", 5)})
    #To remove cold start effect
    W_pt_soft,   t_pt_soft   = benchmark_pytorch(texte, params_torch_softmax)
    W_pt_neg,    t_pt_neg    = benchmark_pytorch(texte, params_torch_neg)

    return {
        "n_pairs": n_pairs,
        "taille_vocab": taille_vocab,
        "n_epochs": n_epochs,
        #"softmax_numpy": {"time": t_soft, "time_per_pair": t_soft, "emb": W_soft},
        #"neg_numpy":     {"time": t_neg,  "time_per_pair": t_neg, "emb": W_neg},
        "pytorch_softmax": {"time": t_pt_soft,   "time_per_pair": t_pt_soft,   "emb": W_pt_soft},
        "pytorch_neg_sampling": {"time": t_pt_neg,    "time_per_pair": t_pt_neg,    "emb": W_pt_neg},
    }
