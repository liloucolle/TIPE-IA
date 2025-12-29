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
# Test 1 : vitesse (temps par paire)
# --------------------------
def test_vitesse_par_paire(texte, params_numpy, params_torch, seed=0):
    _, _, _, _, _, paires = build_dataset(texte, window_size=params_numpy["window_size"])
    n_pairs = max(1, len(paires))
    n_epochs = params_numpy["n_epochs"]

    W_soft, t_soft = benchmark_softmax_numpy(texte, params_numpy, seed=seed)
    W_neg,  t_neg  = benchmark_neg_numpy(texte, {**params_numpy, "K_neg": params_numpy.get("K_neg", 5)}, seed=seed)
    W_pt,   t_pt   = benchmark_pytorch(texte, params_torch, seed=seed)

    return {
        "n_pairs": n_pairs,
        "softmax_numpy": {"time": t_soft, "time_per_pair": t_soft/(n_pairs*n_epochs), "emb": W_soft},
        "neg_numpy":     {"time": t_neg,  "time_per_pair": t_neg /(n_pairs*n_epochs), "emb": W_neg},
        "pytorch":       {"time": t_pt,   "time_per_pair": t_pt  /(n_pairs*params_torch["n_epochs"]), "emb": W_pt},
    }


# --------------------------
# Test 2 : précision (top-k accuracy)
# --------------------------
def test_precision_topk(texte, params_numpy, params_torch, k=5, max_pairs_eval=2000):
    _, _, _, _, _, paires = build_dataset(texte, window_size=params_numpy["window_size"])
    p_eval = paires[:max_pairs_eval]

    W_soft, _ = benchmark_softmax_numpy(texte, params_numpy)
    W_neg,  _ = benchmark_neg_numpy(texte, {**params_numpy, "K_neg": params_numpy.get("K_neg", 5)})
    W_pt,   _ = benchmark_pytorch(texte, params_torch)

    return {
        "softmax_numpy": topk_accuracy(W_soft, p_eval, k=k),
        "neg_numpy":     topk_accuracy(W_neg,  p_eval, k=k),
        "pytorch":       topk_accuracy(W_pt,   p_eval, k=k),
    }


# --------------------------
# Test 3 : convergence (proxy loss vs epochs)
# --------------------------
def _proxy_loss(emb, paires, max_pairs=1000):
    """
    Proxy stable: moyenne de -cos(emb[centre], emb[contexte]) sur un sous-ensemble fixe.
    (Plus bas = mieux)
    """
    if len(paires) == 0:
        return 0.0
    E = normalize_rows(emb)
    p = paires[:max_pairs]
    vals = [-(E[c] @ E[pos]) for (c, pos) in p]
    return float(np.mean(vals))

def test_convergence_proxy(texte, params_numpy, params_torch, n_checkpoints=10):
    _, _, _, _, _, paires = build_dataset(texte, window_size=params_numpy["window_size"])
    p_eval = paires[:1000]

    max_epochs = params_numpy["n_epochs"]
    checkpoints = np.linspace(1, max_epochs, num=min(n_checkpoints, max_epochs), dtype=int)

    soft_curve, neg_curve, pt_curve = [], [], []

    for e in checkpoints:
        p_np = dict(params_numpy)
        p_np["n_epochs"] = int(e)
        p_ng = {**p_np, "K_neg": params_numpy.get("K_neg", 5)}

        p_t = dict(params_torch)
        p_t["n_epochs"] = int(e)

        W_soft, _ = benchmark_softmax_numpy(texte, p_np)
        W_neg,  _ = benchmark_neg_numpy(texte, p_ng)
        W_pt,   _ = benchmark_pytorch(texte, p_t)

        soft_curve.append(_proxy_loss(W_soft, p_eval))
        neg_curve.append(_proxy_loss(W_neg,  p_eval))
        pt_curve.append(_proxy_loss(W_pt,   p_eval))

    return {"epochs": checkpoints.tolist(), "softmax_numpy": soft_curve, "neg_numpy": neg_curve, "pytorch": pt_curve}


# --------------------------
# Test 4 : analogies (king - man + woman -> queen)
# --------------------------
def most_similar_to_vector(emb, itos, vec, topk=20, exclude=None):
    exclude = exclude or set()
    E = normalize_rows(emb)
    v = vec / (np.linalg.norm(vec) + 1e-12)
    sims = E @ v
    idxs = np.argsort(-sims)
    out = []
    for j in idxs:
        w = itos[j]
        if w in exclude:
            continue
        out.append((w, float(sims[j])))
        if len(out) >= topk:
            break
    return out

def test_analogie(texte, params_numpy, params_torch, a="king", b="man", c="woman", target="queen", topk=20):
    _, _, _, stoi, itos, _ = build_dataset(texte, window_size=params_numpy["window_size"])

    def one_model(emb):
        if a not in stoi or b not in stoi or c not in stoi or target not in stoi:
            return {"ok": False, "rank": None, "neighbors": []}
        vec = emb[stoi[a]] - emb[stoi[b]] + emb[stoi[c]]
        neigh = most_similar_to_vector(emb, itos, vec, topk=topk, exclude={a, b, c})
        rank = None
        for i, (w, _) in enumerate(neigh, start=1):
            if w == target:
                rank = i
                break
        return {"ok": True, "rank": rank, "neighbors": neigh}

    W_soft, _ = benchmark_softmax_numpy(texte, params_numpy)
    W_neg,  _ = benchmark_neg_numpy(texte, {**params_numpy, "K_neg": params_numpy.get("K_neg", 5)})
    W_pt,   _ = benchmark_pytorch(texte, params_torch)

    return {"softmax_numpy": one_model(W_soft), "neg_numpy": one_model(W_neg), "pytorch": one_model(W_pt)}


# --------------------------
# Test 5 : sensibilité à K_neg (temps & qualité)
# --------------------------
def test_sensibilite_K(texte, params_numpy, params_torch, K_list=(2,5,10,20), k_topk=5):
    _, _, _, _, _, paires = build_dataset(texte, window_size=params_numpy["window_size"])
    p_eval = paires[:2000]

    out = {"K": [], "neg_numpy_time": [], "neg_numpy_acc": [], "pt_time": [], "pt_acc": []}

    for K in K_list:
        p_np = {**params_numpy, "K_neg": int(K)}
        W_neg, t_neg = benchmark_neg_numpy(texte, p_np)
        acc_neg = topk_accuracy(W_neg, p_eval, k=k_topk)

        p_t = dict(params_torch)
        # côté pytorch, K est généralement interne dans le modèle ; si ton deep_learning_pytorch ne l’a pas,
        # tu ne peux pas le varier. Dans ce cas, laisse constant et commente ce test côté PyTorch.
        # => ici on n'essaie pas de passer K à pytorch.
        W_pt, t_pt = benchmark_pytorch(texte, p_t)
        acc_pt = topk_accuracy(W_pt, p_eval, k=k_topk)

        out["K"].append(int(K))
        out["neg_numpy_time"].append(float(t_neg))
        out["neg_numpy_acc"].append(float(acc_neg))
        out["pt_time"].append(float(t_pt))
        out["pt_acc"].append(float(acc_pt))

    return out


# --------------------------
# Test 6 : scalabilité (temps vs vocab V)
# --------------------------
def _inflate_vocab_by_suffix(texte, n_variants=1):
    """
    Augmente artificiellement V en suffixant les mots (king -> king__0, king__1, ...).
    """
    texte = canon_texte(texte)
    all_versions = []
    for j in range(n_variants):
        suffix = f"__{j}"
        new_lines = []
        for line in texte:
            out = []
            cur = []
            for ch in line:
                if ch.isalpha():
                    cur.append(ch)
                else:
                    if cur:
                        out.append("".join(cur) + suffix)
                        cur = []
                    out.append(ch)
            if cur:
                out.append("".join(cur) + suffix)
            new_lines.append("".join(out))
        all_versions.extend(new_lines)
    return all_versions

def test_scalabilite_vocab(texte_base, params_numpy, params_torch, variants_list=(1,2,4,8)):
    out = {"variants": [], "V": [], "pairs": [], "t_soft": [], "t_neg": [], "t_pt": []}

    for nv in variants_list:
        corpus = _inflate_vocab_by_suffix(texte_base, n_variants=int(nv))
        _, freq, vocab, stoi, itos, paires = build_dataset(corpus, window_size=params_numpy["window_size"])
        V = len(vocab)
        P = len(paires)

        _, t_soft = benchmark_softmax_numpy(corpus, params_numpy)
        _, t_neg  = benchmark_neg_numpy(corpus, {**params_numpy, "K_neg": params_numpy.get("K_neg", 5)})
        _, t_pt   = benchmark_pytorch(corpus, params_torch)

        out["variants"].append(int(nv))
        out["V"].append(int(V))
        out["pairs"].append(int(P))
        out["t_soft"].append(float(t_soft))
        out["t_neg"].append(float(t_neg))
        out["t_pt"].append(float(t_pt))

    return out