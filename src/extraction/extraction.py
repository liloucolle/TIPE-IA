from content.echantillon_texte import (article)
from text_utils.text_utils import tokenisation
import numpy as np

def calcul_vecteur_moyen(embeddings, text):
    """
    Calcule le vecteur moyen des embeddings des mots donnés.
    """
    texte = tokenisation(text)

    vectors = []
    for mot in texte:
        if mot in embeddings:
            vectors.append(embeddings[mot])
    if not vectors:
        return None
    return np.mean(vectors, axis=0)

def top_n_keywords_from_text(embeddings, text, N=10):
    """
    Retourne les top-N mots dont les embeddings ont la plus grande similarité cosinus
    avec le vecteur moyen des mots présents dans `text`.

    Params:
    - embeddings: dict mapping mot -> np.ndarray
    - text: chaîne de texte
    - n: nombre de mots à retourner (par défaut 10)

    Retourne une liste de tuples `(mot, similarite)` triée par similarité décroissante.
    """

    mean_vec = calcul_vecteur_moyen(embeddings, text, N)
    if mean_vec is None:
        return []

    sims = []
    mean_norm = np.linalg.norm(mean_vec)
    if mean_norm == 0:
        return []

    for mot, vec in embeddings.items():
        vec = np.asarray(vec) # Assurer que c'est un tableau NumPy
        denominator = mean_norm * np.linalg.norm(vec)  # Normalisation, 'np.linalg.norm' calcule la norme euclidienne
        if denominator == 0:
            sim = 0.0
        else:
            sim = float(np.dot(mean_vec, vec) / denominator) # Similarité cosinus
        sims.append((mot, sim))

    sims.sort(key=lambda x: x[1], reverse=True) # Tri décroissant
    return sims[:N]


