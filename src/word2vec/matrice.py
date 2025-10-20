import numpy as np
import math as m
from text_utils.text_utils import tokenisation, vocabulaire

# Choix de la dimmension de la matrice
dimension = 100
window_size = 2

# Exemple de texte
texte = 'Word embeddings translate words into vectors, revealing their hidden relationships within a continuous mathematical space.Thanks to them, machines can finally understand that “king” is to “queen” what “man” is to “woman.”'


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

def  softmax (matrice_centre, matrice_contexte, id_mot_centre, word2id):

    v_c = matrice_centre[id_mot_centre]
    ligne,colonne = np.shape(matrice_contexte)

    s_j = [0 for i in range(ligne)]
    somme = 0

    for mot in matrice_contexte():
        id_mot = word2id[mot]
        produit_scalaire = np.dot(mot,v_c)
        s_j[id_mot] = produit_scalaire
        somme += m.exp(produit_scalaire)
    
    probabilite = [0 for i in range(ligne)]

    for mot in matrice_contexte():
        id_mot = word2id[mot]
        prob = m.exp(s_j[id_mot]) / somme
        probabilite[id_mot] = prob
    
    probabilite





        













