import numpy as np
import math as m
from text_utils.text_utils import tokenisation, frequence_mots, vocabulaire, creation_paires



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

    # freq est un dict mot -> comptage (classique)
    for mot in vocab:
        counts[stoi[mot]] = float(freq.get(mot, 0))

    # sécurité : si tout est à 0 (cas bizarre), retour à une distribution uniforme
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
    v_c = matrice_centre[id_centre]         
    grad_v = np.zeros_like(v_c)

    # positif
    u_pos = matrice_contexte[id_pos]         
    score_pos = float(np.dot(u_pos, v_c))
    g_pos = sigmoid(score_pos) - 1.0        

    # grad_v doit utiliser u_pos "avant mise à jour" -> on l'utilise tout de suite
    grad_v += g_pos * u_pos
    matrice_contexte[id_pos] -= taux_apprentissage * (g_pos * v_c)

    # négatifs
    for id_neg in negatifs:
        id_neg = int(id_neg)
        u_neg = matrice_contexte[id_neg]     # vue
        score_neg = float(np.dot(u_neg, v_c))
        g_neg = sigmoid(score_neg)           # étiquette 0

        grad_v += g_neg * u_neg
        matrice_contexte[id_neg] -= taux_apprentissage * (g_neg * v_c)

    matrice_centre[id_centre] -= taux_apprentissage * grad_v
    return matrice_centre, matrice_contexte



def epoche_ns(matrice_centre, matrice_contexte, paires, probs_neg, taux_apprentissage, K=5):
    ordre = np.random.permutation(len(paires))

    # pré-sample un bloc de négatifs
    neg_pool = np.random.choice(len(probs_neg), size=(len(paires), K), replace=True, p=probs_neg)

    for idx, k in enumerate(ordre):
        id_centre, id_pos = paires[k]
        negatifs = neg_pool[idx]

        # si tu veux éviter pos/centre : fais un petit masque rapide (sans while)
        mask = (negatifs == id_pos) | (negatifs == id_centre)
        if np.any(mask):
            negatifs[mask] = np.random.choice(len(probs_neg), size=int(mask.sum()), replace=True, p=probs_neg)

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

    len_pair = len(paires)

    total = 0.0
    for (id_c, id_pos) in paires:
        negatifs = echantillon_negatifs(probs_neg, id_pos=id_pos, K=K, id_centre=id_c)
        total += loss_pair_negative_sampling(matrice_centre, matrice_contexte, id_c, id_pos, negatifs)

    np.random.set_state(rng_state)
    return float(total / len_pair) # moyenne par paire


def entrainer_ns(matrice_centre, matrice_contexte, paires, probs_neg,
                 taux_apprentissage=0.05, n_epochs=10, decay=0.95, K=5):
    """
    Entraîne le modèle Word2Vec avec Negative Sampling sur plusieurs époques.
    
    Paramètres
    ----------
    matrice_centre : np.ndarray
        Matrice des embeddings des mots centraux.
    
    matrice_contexte : np.ndarray
        Matrice des embeddings des mots de contexte.
    
    paires : List[Tuple[int, int]]
        Liste des paires (mot_central, mot_contexte).
    
    probs_neg : np.ndarray
        Vecteur de probabilités pour l'échantillonnage négatif.
    
    taux_apprentissage : float
        Learning rate initial.
    
    n_epochs : int
        Nombre d'époques d'entraînement.
    
    decay : float
        Coefficient de décroissance du learning rate.
    
    K : int
        Nombre d'échantillons négatifs par paire.
    
    Retour
    -------
    Tuple[np.ndarray, np.ndarray]
        Matrices des embeddings entraînés (centre, contexte).
    """
    
    eta = taux_apprentissage

    for epoch in range(n_epochs):
        matrice_centre, matrice_contexte = epoche_ns(
            matrice_centre, matrice_contexte, paires, probs_neg, eta, K=K
        )
        
        L = perte_tot_ns(matrice_centre, matrice_contexte, paires, probs_neg, K=K, seed=epoch + 1)
        print(f"Époque {epoch+1}/{n_epochs}, perte totale (NS): {L:.4f}")

        eta *= decay

    return matrice_centre, matrice_contexte



def deep_learning_negative_sampling(texte, dimension, taux_apprentissage, window_size,
                                   n_epochs, decay, K_neg=5, alpha_unigram=0.75):
    """
    Entraîne un modèle Word2Vec de type Skip-Gram avec Negative Sampling.
    
    Paramètres
    ----------
    texte : str
        Texte brut utilisé pour l'entraînement.
    
    dimension : int
        Dimension des vecteurs d'embedding.
    
    taux_apprentissage : float
        Learning rate initial de la descente de gradient.
    
    window_size : int
        Taille de la fenêtre de contexte autour de chaque mot central.
    
    n_epochs : int
        Nombre d'époques d'entraînement.
    
    decay : float
        Coefficient de décroissance du taux d'apprentissage.
    
    K_neg : int
        Nombre d'échantillons négatifs par paire (par défaut 5).
    
    alpha_unigram : float
        Exponent pour la distribution unigram (par défaut 0.75 comme Word2Vec).
    
    Retour
    -------
    Tuple[np.ndarray, np.ndarray]
        - matrice_centre : embeddings des mots centraux
        - matrice_contexte : embeddings des mots de contexte
    """

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
        decay=decay,
        K=K_neg
    )

    return matrice_centre, matrice_contexte

