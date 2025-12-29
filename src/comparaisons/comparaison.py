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
    # on garde list[str] si on en a une (utile pour pytorch),
    # mais si c’est un str, on le met en list[str] d’une ligne
    if isinstance(texte, list):
        return texte
    return [texte]

def set_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)


def cosine(u, v):
    return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))


def topk_accuracy(emb, paires, k=5):
    """
    Mesure simple et défendable :
    pour chaque (centre, contexte),
    est-ce que le vrai contexte est dans les k plus proches voisins ?
    """
    correct = 0
    for c, pos in paires:
        sims = emb @ emb[c]
        sims = sims / (np.linalg.norm(emb, axis=1) * np.linalg.norm(emb[c]) + 1e-12)
        topk = np.argsort(-sims)[:k]
        if pos in topk:
            correct += 1
    return correct / len(paires)


# =========================
# Benchmarks NUMPY
# =========================

def benchmark_softmax_numpy(texte, params):
    set_seed()
    t0 = time.time()
    Wc, Wt = softmax_numpy(texte, **params)
    t = time.time() - t0
    print(str(t) + ":total time benchamark softmax numpy implementation")
    return Wc, t


def benchmark_neg_numpy(texte, params):
    set_seed()
    t0 = time.time()
    Wc, Wt = deep_learning_negative_sampling(texte, **params)
    t = time.time() - t0
    print(str(t) + ":total time benchamark negative sampling numpy implementation")

    return Wc, t


# =========================
# Benchmarks PYTORCH
# =========================

def benchmark_pytorch(texte, params):
    set_seed()
    t0 = time.time()
    from word2vec_pytorch.word2vec_pytorch import deep_learning_pytorch
    Wc = deep_learning_pytorch(texte, **params)
    t = time.time() - t0
    print(str(t) + ":total time benchmark pytorch implementation")
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