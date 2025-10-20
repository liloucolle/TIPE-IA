# Tokenisation du texte

from collections import Counter
from typing import List, Tuple


def tokenisation (texte):
    mots = []
    n = len(texte)
    mot_courant = []
    for i in range(n):
        num = ord(texte[i])

        #print(mot_courant)

        if num >= 65 and num <= 90:
            mot_courant.append(chr(num + 32))
        
        elif num >= 97 and num <= 122:
            mot_courant.append(chr(num))
        
        else: 
            if len(mot_courant) != 0:
                mots.append(''.join(mot_courant))
                mot_courant = []
    
    if mot_courant!= []:
        mots.append(''.join(mot_courant))
    
    return mots

# Fréquence des mots
# tokens: Liste (pour chaque phrase) de listes de mots
def frequence_mots(tokens: List[List[str]]):
    return Counter(w for sent in tokens for w in sent)

# Construction du vocabulaire, et dictionnaires pour access rapide
def vocabulaire2(frequence_mots) -> Tuple[List[str], dict, dict]:
    vocab = [w for w, c in frequence_mots.items()]
    vocab.sort()
    stoi = {w:i for i,w in enumerate(vocab)}
    itos = {i:w for w,i in stoi.items()}
    return vocab, stoi, itos


# Construction du vocabulaire
def vocabulaire (texte):

    t = tokenisation(texte)
    word2id = {}
    id2word = {}

    compteur = 0

    for i in range(len(t)):
        if word2id.get(t[i]) == None:
            word2id[t[i]] = compteur
            id2word[compteur] = t[i]
            compteur += 1
    

    return word2id,id2word


def creation_paires(tokens: list, word2id: dict, taille_fenetre: int = 1):
    """
    Crée des paires (mot_centre, mot_contexte) pour l'entraînement Word2Vec.
    
    Pour chaque mot dans le corpus, cette fonction génère des paires avec tous
    les mots dans sa fenêtre de contexte (mots voisins à gauche et à droite).
    
    Args:
        tokens (list): Liste de phrases, où chaque phrase est une liste de mots tokenisés
        word2id (dict): Dictionnaire de correspondance mot -> identifiant numérique
        taille_fenetre (int, optional): Taille de la fenêtre de contexte de chaque côté
                                       du mot centre. Par défaut 1.
    
    Returns:
        list: Liste de tuples (id_centre, id_contexte) représentant les paires
              d'entraînement pour le modèle Word2Vec
    
    Example:
        >>> tokens = [["le", "chat", "mange"], ["le", "chien", "dort"]]
        >>> word2id = {"le": 0, "chat": 1, "mange": 2, "chien": 3, "dort": 4}
        >>> paires = creation_paires(tokens, word2id, taille_fenetre=1)
        >>> # Génère des paires comme (1, 0), (1, 2) pour "chat" avec "le" et "mange"
    """
    paires = []
    for sent in tokens:
        n = len(sent)
        for i in range(len(sent)):
            for k in range(1,taille_fenetre+1):

                id_central = word2id[sent[i]]

                if i-k >= 0:
                    id_contexte = word2id[sent[i-k]]
                    paires.append((id_central,id_contexte))


                if i+k <= n-1:
                    id_contexte = word2id[sent[i+k]]
                    paires.append((id_central,id_contexte))
    
    return paires