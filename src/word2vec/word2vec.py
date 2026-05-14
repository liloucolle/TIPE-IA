import numpy as np
from typing import List, Tuple, Dict
from text_utils.text_utils import tokenisation, frequence_mots, vocabulaire, creation_paires

np.random.seed(0)

def softmax(matrice_centre: np.ndarray, matrice_contexte: np.ndarray, id_mot_centre: int) -> np.ndarray:

    v_c = matrice_centre[id_mot_centre]

    s_j = np.dot(matrice_contexte, v_c)          # produit scalaire pour chaque mot
    s_j_xmax = np.max(s_j)
    exp_sj = np.exp(s_j - s_j_xmax)           # enlève le max pour réduire l'exponentielle sans changer la valeur de la probabilité à la fin
    probabilite = exp_sj / np.sum(exp_sj)

    # Vérifier que toutes les probabilités sont entre 0 et 1
    assert np.all((probabilite >= 0) & (probabilite <= 1)), f"Probabilités hors limites: min={np.min(probabilite)}, max={np.max(probabilite)}"
    assert probabilite.ndim == 1, f"Shape inattendu: {probabilite.shape}"

    return probabilite

def descente_de_gradient(matrice_centre: np.ndarray, matrice_contexte: np.ndarray, probabilite: np.ndarray, 
                        id_mot_centre: int, id_mot_context_positif: int, taux_apprentissage: float) -> Tuple[np.ndarray, np.ndarray]:


    lignes = np.shape(matrice_contexte)[0] # nombre total de mots
    y = np.zeros(lignes, dtype= float) # vecteur cible de même forme que probabilité
    y[id_mot_context_positif] = 1.0  # embedding idéal (one hot vector)

    v_c = matrice_centre[id_mot_centre]
    erreur = (probabilite - y) # comparaison de la prédiction de softmax et one hot vector
    grad_centre = np.matmul(matrice_contexte.T, erreur) # calcule les corrections nécessaires
    grad_contexte = np.outer(erreur,v_c) # produit exterieur pour savoir la correction pour chaque vecteur context

    # Amélioration des matrices
    matrice_contexte -= taux_apprentissage * grad_contexte 
    matrice_centre[id_mot_centre] -= taux_apprentissage * grad_centre


    return matrice_centre, matrice_contexte

def epoque(matrice_centre: np.ndarray, matrice_contexte: np.ndarray, paires: List[Tuple[int, int]], taux_apprentissage: float) -> Tuple[np.ndarray, np.ndarray]:

    # On mélange l'ordre des mots central
    ordre = np.random.permutation(len(paires))

    # On boucle sur les mots
    for k in ordre:
        id_centre, id_context = paires[k]
        probabilite = softmax(matrice_centre, matrice_contexte, id_centre)
        matrice_centre, matrice_contexte = descente_de_gradient(matrice_centre, matrice_contexte, probabilite, id_centre, id_context, taux_apprentissage)


    return matrice_centre, matrice_contexte
    

def perte_tot(matrice_centre, matrice_contexte, paires):
    """
    Calcule la fonction de perte moyenne du modèle Skip-Gram avec softmax.

    Principe
    ---------
    Pour chaque paire :
        1. calcul des probabilités via le softmax
        2. récupération de la probabilité du vrai mot contexte
        3. ajout de la perte :
               -log(P(contexte | mot_central))

    La moyenne finale correspond à la fonction de coût du modèle.
    """

    total = 0.0

    for (id_c, id_pos) in paires:

        # Calcul des probabilités softmax pour le mot central
        p = softmax(matrice_centre, matrice_contexte, id_c)

        # Ajout de la cross-entropy loss
        # 1e-12 évite les problèmes numériques liés à log(0)
        total += -np.log(p[id_pos] + 1e-12)

    # Moyenne des pertes sur toutes les paires
    return total / max(1, len(paires))



def entrainer(
    matrice_centre: np.ndarray,
    matrice_contexte: np.ndarray,
    paires: List[Tuple[int, int]],
    taux_apprentissage: float,
    n_epochs: int,
    decay: float
) -> Tuple[np.ndarray, np.ndarray]:

    """
    Entraîne le modèle Skip-Gram sur plusieurs époques.

    Paramètres
    ----------
    matrice_centre : np.ndarray
        Matrice des embeddings des mots centraux.

    matrice_contexte : np.ndarray
        Matrice des embeddings des mots de contexte.

    paires : List[Tuple[int, int]]
        Liste des paires (mot_central, mot_contexte).

    taux_apprentissage : float
        Learning rate initial utilisé pour la descente de gradient.

    n_epochs : int
        Nombre total d'époques d'entraînement.

    decay : float
        Coefficient de décroissance du learning rate.

    Retour
    -------
    Tuple[np.ndarray, np.ndarray]
        Matrices des embeddings entraînés.

    Principe
    ---------
    À chaque époque :
        1. entraînement sur toutes les paires du corpus
        2. calcul de la fonction de perte
        3. affichage de la progression
        4. diminution progressive du learning rate

    Le decay permet :
        - de faire de grandes corrections au début,
        - puis des ajustements plus fins en fin d'entraînement.
    """

    nb_epochs_passe = 0

    for epoch in range(n_epochs):

        print(
            f"Époque {epoch+1}/{n_epochs} "
            f"- Taux d'apprentissage: {taux_apprentissage:.6f}"
        )

        # Effectue une époque complète d'entraînement
        # (mise à jour des embeddings via descente de gradient)
        matrice_centre, matrice_contexte = epoque(
            matrice_centre,
            matrice_contexte,
            paires,
            taux_apprentissage
        )

        # Calcul de la fonction de perte moyenne
        L = perte_tot(
            matrice_centre,
            matrice_contexte,
            paires
        )

        # Mise à jour du nombre d'époques effectuées
        nb_epochs_passe += 1

        # Décroissance progressive du learning rate
        taux_apprentissage *= decay

        # Affichage de la perte
        print(f"Perte moyenne : {L:.6f}")

    return matrice_centre, matrice_contexte

def deep_learning(
    texte: str,
    dimension: int,
    taux_apprentissage: float,
    window_size: int,
    n_epochs: int,
    decay: float
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int], Dict[int, str]]:

    """
    Entraîne un modèle Word2Vec de type Skip-Gram avec softmax.

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

    Retour
    -------
    Tuple[np.ndarray, np.ndarray, Dict[str, int], Dict[int, str]]
        - matrice_centre : embeddings des mots centraux
        - matrice_contexte : embeddings des mots de contexte
        - stoi : dictionnaire mot -> indice
        - itos : dictionnaire indice -> mot

    Principe
    --------
    La fonction :
        1. découpe le texte en tokens
        2. construit le vocabulaire
        3. initialise les matrices d'embeddings
        4. crée les paires (mot central, mot contexte)
        5. entraîne les embeddings par descente de gradient
    """

    # Découpe le texte en mots
    tokens = tokenisation(texte)

    # Si le texte est vide après tokenisation, on renvoie des matrices vides
    if len(tokens[0]) == 0:
        return np.empty((0, dimension)), np.empty((0, dimension))

    # Calcul des fréquences des mots
    freq = frequence_mots(tokens)

    # Construction du vocabulaire et des dictionnaires d'indices
    vocab, stoi, itos = vocabulaire(freq)
    n = len(vocab)

    # Initialisation aléatoire des vecteurs des mots centraux
    matrice_centre = (np.random.rand(n, dimension) - 0.5) / dimension

    # Initialisation à zéro des vecteurs des mots de contexte
    matrice_contexte = np.zeros((n, dimension))

    # Création des paires d'entraînement (mot central, mot contexte)
    paires = creation_paires(tokens, stoi, taille_fenetre=window_size)

    # Entraînement du modèle
    matrice_centre, matrice_contexte = entrainer(
        matrice_centre,
        matrice_contexte,
        paires,
        taux_apprentissage,
        n_epochs,
        decay
    )

    return matrice_centre, matrice_contexte
