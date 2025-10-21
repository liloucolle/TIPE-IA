import numpy as np
import math as m
from text_utils.text_utils import tokenisation, frequence_mots, vocabulaire, creation_paires, cosine_similarity
from content.echantillon_texte import texte_king_and_queen_2 as texte


def softmax(matrice_centre, matrice_contexte, id_mot_centre):

    v_c = matrice_centre[id_mot_centre]

    s_j = np.dot(matrice_contexte, v_c)          # produit scalaire pour chaque mot
    exp_sj = np.exp(s_j - np.max(s_j))           # enlève le max pour réduire l'exponentielle sans changer la valeur de la probabilité à la fin 
    probabilite = exp_sj / np.sum(exp_sj)

    # Vérifier que toutes les probabilités sont entre 0 et 1
    assert np.all((probabilite >= 0) & (probabilite <= 1)), f"Probabilités hors limites: min={np.min(probabilite)}, max={np.max(probabilite)}"

    return probabilite

def renormaliser_matrice(M, borne=1.0):
    """
    Ramène chaque vecteur de la matrice M à une norme au plus égale à 'borne'.

    Args:
        M (np.ndarray): matrice (n_mots × dimension)
        borne (float): norme maximale autorisée (par défaut 1.0)

    Returns:
        np.ndarray: matrice renormalisée
    """
    for i in range(M.shape[0]):
        norme = np.linalg.norm(M[i])
        if norme > borne:
            M[i] = M[i] / norme * borne   # on garde la direction, on réduit la longueur
    return M

def descente_de_gradient(matrice_centre, matrice_contexte, probabilite, id_mot_centre, id_mot_context_positif,taux_apprentissage ):

    lignes = np.shape(matrice_contexte)[0]
    y = np.zeros(lignes, dtype= float) # pour que y soit de la même forme que probabilité
    y[id_mot_context_positif] = 1.0

    grad = np.matmul(matrice_contexte.T, (probabilite - y))

    v_c = matrice_centre[id_mot_centre]
    matrice_contexte -= taux_apprentissage * np.outer((probabilite - y),v_c) # Calcule du produit exterieur 

    matrice_centre[id_mot_centre] = matrice_centre[id_mot_centre] - taux_apprentissage * grad

    matrice_centre = renormaliser_matrice(matrice_centre, borne=1.0)
    matrice_contexte = renormaliser_matrice(matrice_contexte, borne=1.0)


    # Vérifier que les valeurs de matrice_contexte restent entre 0 et 1
    assert np.all((matrice_contexte >= -1) & (matrice_contexte <= 1)), \
        f"matrice_contexte hors limites dans descente_de_gradient: min={np.min(matrice_contexte)}, max={np.max(matrice_contexte)}"
    
    # Vérifier que les valeurs de matrice_centre restent entre 0 et 1
    assert np.all((matrice_centre >= -1) & (matrice_centre <= 1)), \
        f"matrice_centre hors limites dans descente_de_gradient: min={np.min(matrice_centre)}, max={np.max(matrice_centre)}"

    return matrice_centre

def epoche (matrice_centre, matrice_contexte, paires, taux_apprentissage):

    # On mélange l'ordre des mots central
    ordre = np.random.permutation(len(paires))

    # On boucle sur les mots
    for k in ordre:
        id_centre, id_context = paires[k]
        probabilite = softmax(matrice_centre, matrice_contexte, id_centre)
        matrice_centre = descente_de_gradient(matrice_centre, matrice_contexte, probabilite, id_centre, id_context, taux_apprentissage)

    # Vérifier que les valeurs de matrice_centre restent entre -1 et 1
    assert np.all((matrice_centre >= -1) & (matrice_centre <= 1)), \
        f"matrice_centre hors limites dans epoche: min={np.min(matrice_centre)}, max={np.max(matrice_centre)}"

    # Vérifier que les valeurs de matrice_contexte restent entre -1 et 1
    assert np.all((matrice_contexte >= -1) & (matrice_contexte <= 1)), \
        f"matrice_contexte hors limites dans epoche: min={np.min(matrice_contexte)}, max={np.max(matrice_contexte)}"

    return matrice_centre, matrice_contexte
    
def perte_tot(matrice_centre, matrice_contexte, paires):

    total = 0.0

    for (id_c, id_pos) in paires:
        p = softmax(matrice_centre, matrice_contexte, id_c)
        total += -np.log(p[id_pos])

    return float(total)

def entrainer(matrice_centre, matrice_contexte, paires, taux_apprentissage=0.05, n_epochs=10, tol=1e-3, decay=0.01):

    meilleur_loss = perte_tot(matrice_centre, matrice_contexte, paires)

    eta = taux_apprentissage

    for epoch in range(n_epochs):
        # une époque d'apprentissage
        matrice_centre, matrice_contexte = epoche(matrice_centre, matrice_contexte, paires, eta)

        # évaluer la perte
        L = perte_tot(matrice_centre, matrice_contexte, paires)

        # amélioration relative
        rel_gain = (meilleur_loss - L) / max(1e-12, meilleur_loss)

        # décroissance du LR (douce et efficace en Word2Vec)
        eta *= decay

    return matrice_centre, matrice_contexte

def deep_learning(texte, dimension, taux_apprentissage, window_size, n_epochs, decay):

    tokens = tokenisation(texte)

    if len(tokens[0]) == 0:
        return np.empty((0, dimension)), np.empty((0, dimension))

    freq = frequence_mots(tokens)
    vocab, stoi, itos = vocabulaire(freq)
    n = len(vocab)

    matrice_centre = (np.random.rand(n, dimension) - 0.5) / dimension
    matrice_contexte = (np.random.rand(n, dimension) - 0.5) / dimension

    paires = creation_paires(tokens, stoi, taille_fenetre=window_size)

    losses = []
    meilleur_loss = perte_tot(matrice_centre, matrice_contexte, paires)
    losses.append(meilleur_loss)

    aucune_amelioration = 0
    eta = taux_apprentissage

    for epoch in range(n_epochs):
        matrice_centre, matrice_contexte = epoche(
            matrice_centre, matrice_contexte, paires, eta
        )

        L = perte_tot(matrice_centre, matrice_contexte, paires)
        losses.append(L)

        eta *= decay

    return matrice_centre, matrice_contexte, stoi, itos, losses

m_centre, m_context, word2id, id2word, losses = deep_learning(texte, dimension = 100, 
                                                              taux_apprentissage = 0.0005, 
                                                              window_size = 2, 
                                                              n_epochs = 50, 
                                                              Sdecay = 0.99)

id_queen = word2id["queen"]
id_king = word2id["king"]
id_woman = word2id["woman"]
id_man = word2id["man"]


v_queen = m_centre[id_queen]
v_king = m_centre[id_king]
v_woman = m_centre[id_woman]
v_man = m_centre[id_man]


print(losses)
print(cosine_similarity(v_woman, v_queen))
print(cosine_similarity(m_centre[word2id["cup"]], v_queen))

