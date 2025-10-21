import numpy as np
import math as m
from text_utils.text_utils import tokenisation, frequence_mots, vocabulaire, creation_paires, cosine_similarity
from content.echantillon_texte import texte_king_and_queen_2 as texte


def softmax(matrice_centre, matrice_contexte, id_mot_centre):

    v_c = matrice_centre[id_mot_centre]

    s_j = np.dot(matrice_contexte, v_c)          # produit scalaire pour chaque mot
    s_j_xmax = np.max(s_j)
    exp_sj = np.exp(s_j - s_j_xmax)           # enlève le max pour réduire l'exponentielle sans changer la valeur de la probabilité à la fin
    probabilite = exp_sj / np.sum(exp_sj)

    # Vérifier que toutes les probabilités sont entre 0 et 1
    assert np.all((probabilite >= 0) & (probabilite <= 1)), f"Probabilités hors limites: min={np.min(probabilite)}, max={np.max(probabilite)}"
    assert probabilite.ndim == 1, f"Shape inattendu: {probabilite.shape}"

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

    v_c = matrice_centre[id_mot_centre]
    erreur = (probabilite - y)
    grad_centre = np.matmul(matrice_contexte.T, erreur)
    grad_contexte = np.outer(erreur,v_c)

    # Amélioration des matrices
    matrice_contexte -= taux_apprentissage * grad_contexte 
    matrice_centre[id_mot_centre] -= taux_apprentissage * grad_centre


    # Vérifier que les valeurs de matrice_contexte restent entre 0 et 1
    assert np.all((matrice_contexte >= -1) & (matrice_contexte <= 1)), \
        f"matrice_contexte hors limites dans descente_de_gradient: min={np.min(matrice_contexte)}, max={np.max(matrice_contexte)}"
    
    # Vérifier que les valeurs de matrice_centre restent entre 0 et 1
    assert np.all((matrice_centre >= -1) & (matrice_centre <= 1)), \
        f"matrice_centre hors limites dans descente_de_gradient: min={np.min(matrice_centre)}, max={np.max(matrice_centre)}"

    return matrice_centre, matrice_contexte

def epoche (matrice_centre, matrice_contexte, paires, taux_apprentissage):

    # On mélange l'ordre des mots central
    ordre = np.random.permutation(len(paires))

    # On boucle sur les mots
    for k in ordre:
        id_centre, id_context = paires[k]
        probabilite = softmax(matrice_centre, matrice_contexte, id_centre)
        matrice_centre, matrice_contexte = descente_de_gradient(matrice_centre, matrice_contexte, probabilite, id_centre, id_context, taux_apprentissage)

    # Normaliser les matrices pour éviter débordement après
    matrice_centre = renormaliser_matrice(matrice_centre, borne=1.0)
    matrice_contexte = renormaliser_matrice(matrice_contexte, borne=1.0)

    # Vérifier que les valeurs de matrice_centre restent entre -1 et 1
    assert np.all((matrice_centre >= -1) & (matrice_centre <= 1)), \
        f"matrice_centre hors limites dans epoche: min={np.min(matrice_centre)}, max={np.max(matrice_centre)}"

    # Vérifier que les valeurs de matrice_contexte restent entre -1 et 1
    assert np.all((matrice_contexte >= -1) & (matrice_contexte <= 1)), \
        f"matrice_contexte hors limites dans epoche: min={np.min(matrice_contexte)}, max={np.max(matrice_contexte)}"

    return matrice_centre, matrice_contexte
    
def perte_tot(matrice_centre, matrice_contexte, paires):
    """
    Calcule la perte moyenne (log-loss) sur toutes les paires.
    Skip-Gram avec softmax plein.
    """
    total = 0.0
    for (id_c, id_pos) in paires:
        p = softmax(matrice_centre, matrice_contexte, id_c)
        total += -np.log(p[id_pos] + 1e-12)  # stabilité
    return total / max(1, len(paires))

def entrainer(matrice_centre, matrice_contexte, paires, taux_apprentissage, n_epochs, decay):

    meilleur_loss = perte_tot(matrice_centre, matrice_contexte, paires)
    nb_epochs_passe = 0
    eta = taux_apprentissage 

    for epoch in range(n_epochs):
        # une époque d'apprentissage
        matrice_centre, matrice_contexte = epoche(matrice_centre, matrice_contexte, paires, eta)

        # évaluer la perte
        L = perte_tot(matrice_centre, matrice_contexte, paires)

        # amélioration relative
        rel_gain = (meilleur_loss - L) / max(1e-12, meilleur_loss)

        # décroissance du LR (douce et efficace en Word2Vec)
        nb_epochs_passe +=1
        eta = taux_apprentissage * (decay ** nb_epochs_passe)

        print(perte_tot(matrice_centre, matrice_contexte, paires))

    return matrice_centre, matrice_contexte

def deep_learning(texte, dimension, taux_apprentissage, window_size, n_epochs, decay):

    tokens = tokenisation(texte)

    if len(tokens[0]) == 0:
        return np.empty((0, dimension)), np.empty((0, dimension))

    freq = frequence_mots(tokens)
    vocab, stoi, itos = vocabulaire(freq)
    n = len(vocab)

    matrice_centre = (np.random.rand(n, dimension)) * (0.5/ dimension)
    matrice_contexte = (np.random.rand(n, dimension)) * (0.5/ dimension)

    paires = creation_paires(tokens, stoi, taille_fenetre=window_size)

    matrice_centre, matrice_contexte = entrainer(matrice_centre,
                                                 matrice_contexte, 
                                                 paires, 
                                                 taux_apprentissage, 
                                                 n_epochs, 
                                                 decay)
        
    return matrice_centre, matrice_contexte, stoi, itos

def mots_similaires(mot, matrice_embeddings, word2id, top_k=5):
    """
    Trouve les mots les plus similaires à un mot donné en utilisant la similarité cosinus.
    
    Args:
        mot (str): le mot de référence
        matrice_embeddings (np.ndarray): matrice des embeddings (n_mots × dimension)
        word2id (dict): dictionnaire mot -> id
        top_k (int): nombre de mots similaires à retourner (par défaut 5)
    
    Returns:
        list: liste de tuples (mot, similarité) triée par similarité décroissante
    """
    if mot not in word2id:
        print(f"Le mot '{mot}' n'est pas dans le vocabulaire.")
        return []
    
    # Récupérer le vecteur du mot de référence
    vecteur_mot = matrice_embeddings[word2id[mot]]
    
    # Calculer la similarité avec tous les autres mots
    similarites = []
    for autre_mot, idx in word2id.items():
        if autre_mot != mot:  # Exclure le mot lui-même
            vecteur_autre = matrice_embeddings[idx]
            sim = cosine_similarity(vecteur_mot, vecteur_autre)
            similarites.append((autre_mot, float(sim)))
    
    # Trier par similarité décroissante et prendre les top_k
    similarites.sort(key=lambda x: x[1], reverse=True)
    
    return similarites[:top_k]

m_centre, m_context, word2id, id2word = deep_learning(texte, dimension = 100, 
                                                              taux_apprentissage = 0.1, 
                                                              window_size = 1, 
                                                              n_epochs = 100, 
                                                              decay = 0.99)

id_queen = word2id["queen"]
id_king = word2id["king"]
id_woman = word2id["woman"]
id_man = word2id["man"]


v_queen = m_centre[id_queen]
v_king = m_centre[id_king]
v_woman = m_centre[id_woman]
v_man = m_centre[id_man]

print(cosine_similarity(v_man, v_queen))
print(cosine_similarity(m_centre[word2id["cup"]], v_queen))

# Test d'analogie: king - man + woman ≈ queen
analogie_vector = v_king - v_man + v_woman
print(f"\nTest d'analogie: king - man + woman")
print(f"Similarité avec queen: {cosine_similarity(analogie_vector, v_queen):.4f}")
print(f"Similarité avec king: {cosine_similarity(analogie_vector, v_king):.4f}")

# Trouvez le mot le plus proche de l'analogie
max_sim = -1
mot_plus_proche = ""
for mot, idx in word2id.items():
    if mot not in ["king", "man", "woman"]:  # Exclure les mots utilisés dans l'analogie
        sim = cosine_similarity(analogie_vector, m_centre[idx])
        if sim > max_sim:
            max_sim = sim
            mot_plus_proche = mot

print(f"Mot le plus proche de 'king - man + woman': {mot_plus_proche} (similarité: {max_sim:.4f})")

# Test de la fonction mots_similaires
print("\n" + "="*50)
print("TEST DE LA FONCTION MOTS_SIMILAIRES")
print("="*50)

mots_a_tester = ["king", "queen", "man", "woman"]
for mot in mots_a_tester:
    if mot in word2id:
        print(f"\nMots les plus similaires à '{mot}':")
        similaires = mots_similaires(mot, m_centre, word2id, top_k=20)
        for i, (mot_sim, sim) in enumerate(similaires, 1):
            print(f"  {i}. {mot_sim}: {sim:.4f}")
    else:
        print(f"\nLe mot '{mot}' n'est pas dans le vocabulaire.")