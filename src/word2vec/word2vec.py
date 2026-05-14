import numpy as np
import math as m
from typing import List, Tuple, Dict, Optional
from text_utils.text_utils import tokenisation, frequence_mots, vocabulaire, creation_paires, cosine_similarity
from content.echantillon_texte import texte_king_and_queen_2 as texte
from embedding_utils.embedding_utils import mots_similaires_numpy, most_similar_to_vector_numpy

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
    

def fonction_perte(matrice_centre, matrice_contexte, paires):
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



def entrainer(matrice_centre: np.ndarray, matrice_contexte: np.ndarray, paires: List[Tuple[int, int]], 
              taux_apprentissage: float, n_epochs: int, decay: float) -> Tuple[np.ndarray, np.ndarray]:

    nb_epochs_passe = 0

    for epoch in range(n_epochs):
        print(f"Époque {epoch+1}/{n_epochs} - Taux d'apprentissage: {taux_apprentissage:.6f}")
        # une époque d'apprentissage
        matrice_centre, matrice_contexte = epoque(matrice_centre, matrice_contexte, paires, taux_apprentissage)

        # évaluer la perte
        L = perte_tot(matrice_centre, matrice_contexte, paires)

        # décroissance du LR (douce et efficace en Word2Vec)
        nb_epochs_passe +=1

        print(perte_tot(matrice_centre, matrice_contexte, paires))

    return matrice_centre, matrice_contexte

def word2vec(texte: str, dimension: int, taux_apprentissage: float, window_size: int, 
             n_epochs: int, decay: float) -> Tuple[np.ndarray, np.ndarray, Dict[str, int], Dict[int, str]]:

    tokens = tokenisation(texte)

    if len(tokens[0]) == 0:
        return np.empty((0, dimension)), np.empty((0, dimension))

    freq = frequence_mots(tokens)
    vocab, stoi, itos = vocabulaire(freq)
    n = len(vocab)

    matrice_centre = (np.random.rand(n, dimension)- 0.5) / dimension
    matrice_contexte = np.zeros((n, dimension))  # Initialisation à zéro

    paires = creation_paires(tokens, stoi, taille_fenetre=window_size)

    matrice_centre, matrice_contexte = entrainer(matrice_centre,
                                                 matrice_contexte, 
                                                 paires, 
                                                 taux_apprentissage, 
                                                 n_epochs, 
                                                 decay)
        
    return matrice_centre, matrice_contexte, stoi, itos

# ===== ENTRAÎNEMENT ET TESTS =====

m_centre, m_context, word2id, id2word = word2vec(texte, dimension = 100, 
                                                        taux_apprentissage = 0.01, 
                                                        window_size = 1, 
                                                        n_epochs = 300, 
                                                        decay = 1)

#
# ===== VERIFICATION DU MODELE =====
#

# Test de la fonction mots_similaires
print("\n" + "="*50)
print("TEST DE SIMILARITES")
print("="*50)

for mot in ["king", "queen", "man", "woman"]:
    if mot in word2id:
        print(f"\nMots les plus similaires à '{mot}':")
        similaires = mots_similaires_numpy(mot, m_centre, word2id, top_k=10)
        for i, (mot_sim, sim) in enumerate(similaires, 1):
            print(f"  {i}. {mot_sim}: {sim:.4f}")
    else:
        print(f"\nLe mot '{mot}' n'est pas dans le vocabulaire.")

# Test d'analogie: king - man + woman ≈ queen
print("\n" + "="*50)
print("TEST D'ANALOGIE AVANCÉ")
print("="*50)

# Calculer le vecteur d'analogie: king - man + woman ≈ queen
if "king" in word2id and "man" in word2id and "woman" in word2id:
    vector_analogie = (m_centre[word2id["king"]] - 
                      m_centre[word2id["man"]] + 
                      m_centre[word2id["woman"]])
    
    # Utiliser la nouvelle fonction pour trouver les mots les plus proches
    exclude_words = ["king", "man", "woman"]  # Exclure les mots de l'analogie
    result = most_similar_to_vector_numpy(vector_analogie, m_centre, word2id, 
                                        exclude_words=exclude_words, top_k=10)
    
    print(f"Analogie: king - man + woman ≈ ?")
    print(f"\nTop 10 résultats:")
    for i, (mot, sim) in enumerate(result, 1):
        marker = " ✓" if mot == "queen" else ""
        print(f"{i:2d}. {mot:<15} (similarité: {sim:.4f}){marker}")
    
    # Chercher le rang de "queen" dans les résultats
    queen_rank = None
    for i, (mot, sim) in enumerate(result, 1):
        if mot == "queen":
            queen_rank = i
            break
    
    if queen_rank:
        print(f"\n🎯 'queen' trouvée au rang {queen_rank} !")
    else:
        print(f"\n❌ 'queen' non trouvée dans le top 10")
else:
    print("❌ Certains mots de l'analogie ne sont pas dans le vocabulaire")

