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
    return Wc, t


def benchmark_neg_numpy(texte, params):
    set_seed()
    t0 = time.time()
    Wc, Wt = deep_learning_negative_sampling(texte, **params)
    t = time.time() - t0
    return Wc, t


# =========================
# Benchmarks PYTORCH
# =========================

def benchmark_pytorch(texte, mode="neg", epochs=10, emb_dim=100):
    set_seed()

    tokens = tokenisation(texte)
    freq = frequence_mots(tokens)
    vocab, stoi, itos = vocabulaire(freq)
    pairs = creation_paires(tokens, stoi, taille_fenetre=2)
    V = len(vocab)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SkipGram(V, emb_dim)

    t0 = time.time()
    if mode == "softmax":
        ds = SkipGramSoftmaxDS(pairs)
        train_softmax(model, ds, epochs=epochs, device=device)
    else:
        unigram = torch.tensor([freq[itos[i]] for i in range(V)], dtype=torch.float)
        unigram = unigram.pow(0.75)
        unigram /= unigram.sum()
        ds = SkipGramNegSampleDS(pairs, V, unigram_probs=unigram, K=5)
        train_neg(model, ds, epochs=epochs, device=device)

    t = time.time() - t0
    emb = model.get_embeddings().numpy()
    return emb, t, pairs



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