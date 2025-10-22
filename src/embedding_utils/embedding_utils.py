"""
Utilitaires pour l'analyse et le test des word embeddings.

Ce module fournit des fonctions spécifiques pour NumPy et PyTorch
pour analyser la qualité des embeddings de mots.

Conventions de nommage :
- Fonctions suffixées par '_numpy' : travaillent avec np.ndarray
- Fonctions suffixées par '_torch' : travaillent avec torch.Tensor
- Fonctions sans suffixe : fonctions génériques/utilitaires
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Union
from text_utils.text_utils import cosine_similarity

# Support optionnel pour PyTorch
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ============================================================================
# FONCTIONS NUMPY
# ============================================================================

def mots_similaires_numpy(mot: str, matrice_embeddings: np.ndarray, 
                         word2id: Dict[str, int], top_k: int = 5) -> List[Tuple[str, float]]:
    """
    Trouve les mots les plus similaires à un mot donné (version NumPy).
    
    Args:
        mot (str): le mot de référence
        matrice_embeddings (np.ndarray): matrice des embeddings (n_mots × dimension)
        word2id (dict): dictionnaire mot -> id
        top_k (int): nombre de mots similaires à retourner
    
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


def most_similar_to_vector_numpy(vector: np.ndarray, matrice_embeddings: np.ndarray, 
                                 word2id: Dict[str, int], exclude_words: Optional[List[str]] = None,
                                 top_k: int = 5) -> List[Tuple[str, float]]:
    """
    Trouve les mots les plus similaires à un vecteur donné (version NumPy).
    
    Args:
        vector (np.ndarray): vecteur cible pour lequel trouver les mots similaires
        matrice_embeddings (np.ndarray): matrice des embeddings (n_mots × dimension)
        word2id (dict): dictionnaire mot -> id
        exclude_words (list, optional): liste de mots à exclure des résultats
        top_k (int): nombre de mots similaires à retourner
    
    Returns:
        list: liste de tuples (mot, similarité) triée par similarité décroissante
    """
    if exclude_words is None:
        exclude_words = []
    
    exclude_words_set = set(exclude_words)
    similarities = []
    
    # Calculer la similarité avec tous les mots (sauf ceux à exclure)
    for mot, idx in word2id.items():
        if mot not in exclude_words_set:
            sim = cosine_similarity(vector, matrice_embeddings[idx])
            similarities.append((mot, float(sim)))
    
    # Trier par similarité décroissante et prendre les top_k
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]


# ============================================================================
# FONCTIONS PYTORCH
# ============================================================================

def mots_similaires_torch(mot: str, matrice_embeddings: 'torch.Tensor', 
                          word2id: Dict[str, int], top_k: int = 5) -> List[Tuple[str, float]]:
    """
    Trouve les mots les plus similaires à un mot donné (version PyTorch optimisée).
    
    Args:
        mot (str): le mot de référence
        matrice_embeddings (torch.Tensor): matrice des embeddings (n_mots × dimension)
        word2id (dict): dictionnaire mot -> id
        top_k (int): nombre de mots similaires à retourner
    
    Returns:
        list: liste de tuples (mot, similarité) triée par similarité décroissante
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch n'est pas disponible")
    
    if mot not in word2id:
        print(f"Le mot '{mot}' n'est pas dans le vocabulaire.")
        return []
    
    # Récupérer le vecteur du mot de référence
    widx = word2id[mot]
    vecteur_mot = matrice_embeddings[widx]
    
    # Calcul vectorisé optimisé avec PyTorch
    sims = F.cosine_similarity(matrice_embeddings, vecteur_mot.unsqueeze(0))  # (V,)
    vals, idxs = torch.topk(sims, k=top_k+1)
    
    # Créer la liste résultat en excluant le mot lui-même
    out = []
    id2word = {idx: mot for mot, idx in word2id.items()}  # Inversion rapide
    
    for v, i in zip(vals.tolist(), idxs.tolist()):
        if i == widx:  # ignorer le mot lui-même
            continue
        if i in id2word:
            out.append((id2word[i], float(v)))
            if len(out) == top_k:
                break
    
    return out


def most_similar_to_vector_torch(vector: 'torch.Tensor', matrice_embeddings: 'torch.Tensor', 
                                 word2id: Dict[str, int], topk: int = 5) -> List[Tuple[str, float]]:
    """
    Trouve les mots les plus similaires à un vecteur donné (version PyTorch optimisée).
    
    Args:
        vector (torch.Tensor): vecteur cible
        matrice_embeddings (torch.Tensor): matrice des embeddings
        word2id (dict): dictionnaire mot -> id
        topk (int): nombre de mots à retourner
    
    Returns:
        list: liste de tuples (mot, similarité)
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch n'est pas disponible")
    
    # Calcul vectorisé optimisé avec PyTorch
    sims = F.cosine_similarity(matrice_embeddings, vector.unsqueeze(0))  # (V,)
    vals, idxs = torch.topk(sims, k=topk)
    
    # Créer la liste résultat
    id2word = {idx: mot for mot, idx in word2id.items()}  # Inversion rapide
    out = []
    
    for v, i in zip(vals.tolist(), idxs.tolist()):
        if i in id2word:
            out.append((id2word[i], float(v)))
    
    return out[:topk]
