import numpy as np
import math as m
from text_utils.text_utils import tokenisation, vocabulaire

# Choix de la dimmension de la matrice
dimension = 100
window_size = 2

# Initialisation des matrices centre et contexte

def intialisation_matrice (texte):
    word2id = vocabulaire(texte)[0]
    n = len(word2id)

    # On enlève 0.5 pour avoir des valeurs entre -0.5 et 0.5
    matrice_centre = np.random.rand(n,dimension) - 0.5
    matrice_contexte = np.random.rand(n,dimension) - 0.5 

    return matrice_centre, matrice_contexte

# Création des paires centre-contexte

def creation_paires (texte, window):
    t = tokenisation(texte)
    word2id = vocabulaire(texte)[0]
    n = len(t)

    paires = []

    for i in range(n):
        for k in range(1,window+1):

            id_central = word2id[t[i]]

            if i-k >= 0:
                id_contexte = word2id[t[i-k]]
                paires.append((id_central,id_contexte))


            if i+k <= n-1:
                id_contexte = word2id[t[i+k]]
                paires.append((id_central,id_contexte))
    
    return paires


def softmax(matrice_centre, matrice_contexte, id_mot_centre):

    v_c = matrice_centre[id_mot_centre]

    s_j = np.dot(matrice_contexte, v_c)          # produit scalaire pour chaque mot
    exp_sj = np.exp(s_j - np.max(s_j))           # enlève le max pour réduire l'exponentielle sans changer la valeur de la probabilité à la fin 
    probabilite = exp_sj / np.sum(exp_sj)

    return probabilite


def fonction_de_perte(probabilite,id_mot_context_positif):

    loss = -m.log(probabilite[id_mot_context_positif])

    return loss

















