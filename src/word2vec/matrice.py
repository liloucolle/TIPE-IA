import numpy as np
import math as m
from text_utils.text_utils import tokenisation, vocabulaire


dimension = 100
window_size = 2
taux_apprentissage = 0.025

# Initialisation des matrices centre et contexte

def intialisation_matrice (texte):
    word2id = vocabulaire(texte)[0]
    n = len(word2id)

    # On enlève 0.5 pour avoir des valeurs entre -0.5 et 0.5
    matrice_centre = np.random.rand(n,dimension) - 0.5
    matrice_contexte = np.random.rand(n,dimension) - 0.5 

    return matrice_centre, matrice_contexte



def softmax(matrice_centre, matrice_contexte, id_mot_centre):

    v_c = matrice_centre[id_mot_centre]

    s_j = np.dot(matrice_contexte, v_c)          # produit scalaire pour chaque mot
    exp_sj = np.exp(s_j - np.max(s_j))           # enlève le max pour réduire l'exponentielle sans changer la valeur de la probabilité à la fin 
    probabilite = exp_sj / np.sum(exp_sj)

    return probabilite


def fonction_de_perte(probabilite,id_mot_context_positif):

    loss = -m.log(probabilite[id_mot_context_positif])

    return loss

def descente_de_gradient(matrice_centre, matrice_contexte, probabilite, id_mot_centre, id_mot_context_positif,taux_apprentissage ):

    lignes = np.shape(matrice_contexte)[0]
    y = np.zeros(lignes, dtype= float) # pour que y soit de la même forme que probabilité
    y[id_mot_context_positif] = 1.0

    grad = np.matmul(matrice_contexte.T, (probabilite - y))

    matrice_centre[id_mot_centre] = matrice_centre[id_mot_centre] - taux_apprentissage * grad

    v_c = matrice_centre[id_mot_centre]
    matrice_contexte -= taux_apprentissage * np.outer((probabilite - y),v_c) # Calcule du produit exterieur 
    
    return matrice_centre

