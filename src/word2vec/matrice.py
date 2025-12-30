import numpy as np
import math as m
from text_utils.text_utils import tokenisation, frequence_mots, vocabulaire, creation_paires
from content.echantillon_texte import texte_100


dimension = 100
window_size = 2
taux_apprentissage = 0.05



def softmax(matrice_centre, matrice_contexte, id_mot_centre):

    v_c = matrice_centre[id_mot_centre]

    s_j = np.dot(matrice_contexte, v_c)          # produit scalaire pour chaque mot
    exp_sj = np.exp(s_j - np.max(s_j))           # enlève le max pour réduire l'exponentielle sans changer la valeur de la probabilité à la fin 
    probabilite = exp_sj / np.sum(exp_sj)

    return probabilite


def descente_de_gradient(matrice_centre, matrice_contexte, probabilite, id_mot_centre, id_mot_context_positif,taux_apprentissage ):

    lignes = np.shape(matrice_contexte)[0]
    y = np.zeros(lignes, dtype= float) # pour que y soit de la même forme que probabilité
    y[id_mot_context_positif] = 1.0

    grad = np.matmul(matrice_contexte.T, (probabilite - y))

    v_c = matrice_centre[id_mot_centre]
    matrice_contexte -= taux_apprentissage * np.outer((probabilite - y),v_c) # Calcule du produit exterieur 

    matrice_centre[id_mot_centre] = matrice_centre[id_mot_centre] - taux_apprentissage * grad


    
    return matrice_centre

def epoche (matrice_centre, matrice_contexte, paires, taux_apprentissage):

    # On mélange l'ordre des mots central

    ordre = np.random.permutation(len(paires))


    # On boucle sur les mots
    for k in ordre:
        id_centre, id_context = paires[k]
        probabilite = softmax(matrice_centre, matrice_contexte, id_centre)
        matrice_centre = descente_de_gradient(matrice_centre, matrice_contexte, probabilite, id_centre, id_context, taux_apprentissage)

    return matrice_centre, matrice_contexte
    
def perte_tot(matrice_centre, matrice_contexte, paires):

    total = 0.0

    for (id_c, id_pos) in paires:
        p = softmax(matrice_centre, matrice_contexte, id_c)
        total += -np.log(p[id_pos])

    return float(total)

def entrainer(matrice_centre, matrice_contexte, paires, taux_apprentissage=0.05, n_epochs=10, patience=3, tol=1e-3, decay=0.95):

    meilleur_loss = perte_tot(matrice_centre, matrice_contexte, paires)

    aucune_amelioration = 0
    eta = taux_apprentissage

    for epoch in range(n_epochs):
        # une époque d'apprentissage
        matrice_centre, matrice_contexte = epoche(matrice_centre, matrice_contexte, paires, eta)

        # évaluer la perte
        L = perte_tot(matrice_centre, matrice_contexte, paires)
        print(f"Époque {epoch+1}, perte totale: {L:.4f}")

        # amélioration relative
        rel_gain = (meilleur_loss - L) / max(1e-12, meilleur_loss)

        # if rel_gain > tol:
        #     meilleur_loss = L
        #     aucune_amelioration = 0

        # else:
        #     aucune_amelioration += 1
        #     if aucune_amelioration >= patience:
        #         break  # early stopping

        # décroissance du LR (douce et efficace en Word2Vec)
        eta *= decay

    return matrice_centre, matrice_contexte

def deep_learning(texte, dimension, taux_apprentissage, window_size, n_epochs, patience, tolerance, decay):


    tokens = tokenisation(texte)

    # Cas bord : corpus vide
    if len(tokens) == 0 or all(len(t) == 0 for t in tokens):
        return np.empty((0, dimension)), np.empty((0, dimension))

    freq = frequence_mots(tokens)
    vocab, stoi, itos = vocabulaire(freq)
    n = len(vocab)

    # 3) Initialisation des matrices centre/contexte dans [-0.5, 0.5]
    matrice_centre = np.random.rand(n, dimension) - 0.5
    matrice_contexte = np.random.rand(n, dimension) - 0.5

    paires = creation_paires(tokens, stoi, taille_fenetre=window_size)

    # 5) Entraînement (épocs + early stopping)
    matrice_centre, matrice_contexte = entrainer(
        matrice_centre,
        matrice_contexte,
        paires,
        taux_apprentissage=taux_apprentissage,
        n_epochs=n_epochs,
        patience=patience,
        tol=tolerance,
        decay=decay,
    )

    return matrice_centre, matrice_contexte

