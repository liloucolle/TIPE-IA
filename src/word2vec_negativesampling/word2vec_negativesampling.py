import numpy as np
import math as m
from text_utils.text_utils import tokenisation, frequence_mots, vocabulaire, creation_paires
from content.echantillon_texte import test_simple


dimension = 100
window_size = 2
taux_apprentissage = 0.05
K_neg = 5              # nombre de négatifs par paire
alpha_unigram = 0.75   # exponent pour la distribution unigram^0.75


def sigmoid(x):
    # stable numériquement
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def distribution_negative_sampling(freq, vocab, stoi, alpha=0.75):
    """
    Construit p(w) ∝ count(w)^alpha (classique Word2Vec: alpha=0.75)
    Retourne un vecteur de proba de taille |V|.
    """
    n = len(vocab)
    counts = np.zeros(n, dtype=float)

    # freq est un dict mot -> count (classique)
    for mot in vocab:
        counts[stoi[mot]] = float(freq.get(mot, 0))

    # sécurité: si tout est à 0 (cas bizarre), fallback uniforme
    if np.sum(counts) == 0:
        return np.ones(n, dtype=float) / n

    probs = np.power(counts, alpha)
    s = np.sum(probs)
    if s == 0:
        return np.ones(n, dtype=float) / n

    return probs / s


def echantillon_negatifs(probs, id_pos, K, id_centre=None):

    n = len(probs)
    negs = np.random.choice(n, size=K, replace=True, p=probs)

    if id_centre is None:
        mask = (negs == id_pos)
    else:
        mask = (negs == id_pos) | (negs == id_centre)

    if np.any(mask):
        negs[mask] = np.random.choice(n, size=int(mask.sum()), replace=True, p=probs)

        # deuxième passe optionnelle si vraiment tu veux être strict
        if id_centre is None:
            mask2 = (negs == id_pos)
        else:
            mask2 = (negs == id_pos) | (negs == id_centre)

        if np.any(mask2):
            negs[mask2] = np.random.choice(n, size=int(mask2.sum()), replace=True, p=probs)

    return negs



def loss_pair_negative_sampling(matrice_centre, matrice_contexte, id_centre, id_pos, negatifs):
    """
    L = -log σ(u_pos·v_c) - Σ log σ(-u_neg·v_c)
    """
    v_c = matrice_centre[id_centre]

    u_pos = matrice_contexte[id_pos]
    score_pos = float(np.dot(u_pos, v_c))
    L = -m.log(sigmoid(score_pos) + 1e-12)

    for id_neg in negatifs:
        u_neg = matrice_contexte[int(id_neg)]
        score_neg = float(np.dot(u_neg, v_c))
        L += -m.log(sigmoid(-score_neg) + 1e-12)

    return float(L)


def descente_de_gradient_ns(matrice_centre, matrice_contexte, id_centre, id_pos, negatifs, taux_apprentissage):
    """
    Updates SGD pour Negative Sampling (skip-gram).
    - centre: v_c
    - contexte: u_w
    """
    v_c = matrice_centre[id_centre]          # vue, pas copy
    grad_v = np.zeros_like(v_c)

    # positif
    u_pos = matrice_contexte[id_pos]         # vue
    score_pos = float(np.dot(u_pos, v_c))
    g_pos = sigmoid(score_pos) - 1.0

    # on stocke u_pos AVANT update pour grad_v
    grad_v += g_pos * u_pos
    matrice_contexte[id_pos] -= taux_apprentissage * (g_pos * v_c)

    # négatifs (boucle sur K, mais sans copy)
    for id_neg in negatifs:
        id_neg = int(id_neg)
        u_neg = matrice_contexte[id_neg]     # vue
        score_neg = float(np.dot(u_neg, v_c))
        g_neg = sigmoid(score_neg)           # label 0

        grad_v += g_neg * u_neg
        matrice_contexte[id_neg] -= taux_apprentissage * (g_neg * v_c)

    matrice_centre[id_centre] -= taux_apprentissage * grad_v
    return matrice_centre, matrice_contexte



def epoche_ns(matrice_centre, matrice_contexte, paires, probs_neg, taux_apprentissage, K=5):
    ordre = np.random.permutation(len(paires))

    for k in ordre:
        id_centre, id_pos = paires[k]
        negatifs = echantillon_negatifs(probs_neg, id_pos=id_pos, K=K, id_centre=id_centre)
        matrice_centre, matrice_contexte = descente_de_gradient_ns(
            matrice_centre, matrice_contexte, id_centre, id_pos, negatifs, taux_apprentissage
        )

    return matrice_centre, matrice_contexte


def perte_tot_ns(matrice_centre, matrice_contexte, paires, probs_neg, K=5, seed=0):
    """
    Approx de la perte totale: on re-sample des négatifs pour chaque paire.
    (utile pour suivre la tendance, pas un "vrai" total déterministe)
    """
    rng_state = np.random.get_state()
    np.random.seed(seed)

    total = 0.0
    for (id_c, id_pos) in paires:
        negatifs = echantillon_negatifs(probs_neg, id_pos=id_pos, K=K, id_centre=id_c)
        total += loss_pair_negative_sampling(matrice_centre, matrice_contexte, id_c, id_pos, negatifs)

    np.random.set_state(rng_state)
    return float(total)


def entrainer_ns(matrice_centre, matrice_contexte, paires, probs_neg,
                 taux_apprentissage=0.05, n_epochs=10, patience=3, tol=1e-3,
                 decay=0.95, K=5, eval_every=0):
    """
    eval_every=0  => AUCUNE évaluation (benchmark temps pur)
    eval_every=5  => évalue toutes les 5 époques
    """

    # ✅ si pas d'évaluation, on évite totalement perte_tot_ns
    if eval_every and eval_every > 0:
        meilleur_loss = perte_tot_ns(matrice_centre, matrice_contexte, paires, probs_neg, K=K, seed=0)
    else:
        meilleur_loss = None

    aucune_amelioration = 0
    eta = taux_apprentissage

    for epoch in range(n_epochs):
        matrice_centre, matrice_contexte = epoche_ns(
            matrice_centre, matrice_contexte, paires, probs_neg, eta, K=K
        )

        # ✅ évaluation optionnelle
        if eval_every and eval_every > 0 and ((epoch + 1) % eval_every == 0):
            L = perte_tot_ns(matrice_centre, matrice_contexte, paires, probs_neg, K=K, seed=epoch + 1)
            rel_gain = (meilleur_loss - L) / max(1e-12, meilleur_loss)

            if rel_gain > tol:
                meilleur_loss = L
                aucune_amelioration = 0
            else:
                aucune_amelioration += 1
                if aucune_amelioration >= patience:
                    break

        eta *= decay

    return matrice_centre, matrice_contexte



def deep_learning_negative_sampling(texte, dimension, taux_apprentissage, window_size,
                                   n_epochs, patience, tolerance, decay,
                                   K_neg=5, alpha_unigram=0.75, eval_every=0):


    tokens = tokenisation(texte)

    if len(tokens[0]) == 0:
        return np.empty((0, dimension)), np.empty((0, dimension))

    freq = frequence_mots(tokens)
    vocab, stoi, itos = vocabulaire(freq)
    n = len(vocab)

    matrice_centre = np.random.rand(n, dimension) - 0.5
    matrice_contexte = np.random.rand(n, dimension) - 0.5

    paires = creation_paires(tokens, stoi, taille_fenetre=window_size)

    probs_neg = distribution_negative_sampling(freq, vocab, stoi, alpha=alpha_unigram)

    matrice_centre, matrice_contexte = entrainer_ns(
        matrice_centre,
        matrice_contexte,
        paires,
        probs_neg,
        taux_apprentissage=taux_apprentissage,
        n_epochs=n_epochs,
        patience=patience,
        tol=tolerance,
        decay=decay,
        K=K_neg, 
        eval_every=0
    )

    return matrice_centre, matrice_contexte

if __name__ == "__main__":
    matrice_centre, matrice_contexte = deep_learning_negative_sampling(
        test_simple,
        dimension=100,
        taux_apprentissage=0.05,
        window_size=2,
        n_epochs=5,
        patience=2,
        tolerance=1e-4,
        decay=0.95,
        K_neg=5,
        alpha_unigram=0.75
    )

    print("\nShape centre :", matrice_centre.shape)
    print("Shape contexte :", matrice_contexte.shape)

    print("\nMatrice embeddings (centre) :")
    print(matrice_centre)