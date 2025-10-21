
import numpy as np
from collections import Counter
from typing import List, Tuple


def tokenisation(texte):
    """
    Tokenise un texte en extrayant uniquement les caractères alphabétiques.
    
    Cette fonction parcourt le texte caractère par caractère, convertit les majuscules
    en minuscules, et sépare les mots en utilisant tous les caractères non-alphabétiques
    comme délimiteurs (espaces, ponctuation, chiffres, etc.).
    
    Args:
        texte (str): Le texte à tokeniser
    
    Returns:
        list: Liste des mots extraits, en minuscules, sans caractères spéciaux
    
    Example:
        >>> tokenisation("Bonjour, monde! Comment ça va?")
        ['bonjour', 'monde', 'comment', 'va']
    """
    texte_token = []
    phrase_courante = []
    mot_courant = []

    for sentence in texte:

        phrase_courante = []
        mot_courant = []

        for i in range(len(sentence)):
            num = ord(sentence[i])

            #print(mot_courant)

            if num >= 65 and num <= 90:
                mot_courant.append(chr(num + 32))
            
            elif num >= 97 and num <= 122:
                mot_courant.append(chr(num))
            
            else: 
                if len(mot_courant) != 0:
                    phrase_courante.append(''.join(mot_courant))
                    mot_courant = []
    
        if mot_courant!= []:
            phrase_courante.append(''.join(mot_courant))
        
        texte_token.append(phrase_courante)

        
    
    return texte_token

def frequence_mots(tokens: List[List[str]]):
    """
    Calcule la fréquence d'apparition de chaque mot dans un corpus tokenisé.
    
    Cette fonction prend une liste de phrases (où chaque phrase est une liste de mots)
    et compte le nombre total d'occurrences de chaque mot unique dans tout le corpus.
    
    Args:
        tokens (List[List[str]]): Liste de phrases, où chaque phrase est une liste de mots tokenisés
    
    Returns:
        Counter: Objet Counter contenant les mots comme clés et leurs fréquences comme valeurs
    
    Example:
        >>> tokens = [["le", "chat", "mange"], ["le", "chien", "mange", "aussi"]]
        >>> freq = frequence_mots(tokens)
        >>> freq["le"]  # Retourne 2
        >>> freq["mange"]  # Retourne 2
    """
    return Counter(w for sent in tokens for w in sent)


def vocabulaire(frequence_mots) -> Tuple[List[str], dict, dict]:
    """
    Construit un vocabulaire ordonné et des dictionnaires de correspondance (version alternative).
    
    À partir d'un objet Counter contenant les fréquences des mots, cette fonction
    crée un vocabulaire trié alphabétiquement et génère deux dictionnaires pour
    faciliter la conversion entre mots et indices numériques en utilisant une
    approche itérative avec boucle for.
    
    Args:
        frequence_mots (Counter): Objet Counter avec les mots et leurs fréquences
    
    Returns:
        Tuple[List[str], dict, dict]: Un tuple contenant :
            - vocab (List[str]): Liste des mots triés alphabétiquement
            - word2id (dict): Dictionnaire mot -> index (alternative à stoi)
            - id2word (dict): Dictionnaire index -> mot (alternative à itos)
    
    Note:
        Cette fonction est équivalente à vocabulaire() mais utilise une approche
        différente pour construire les dictionnaires de correspondance avec des
        noms de variables différents (word2id/id2word au lieu de stoi/itos).
    
    Example:
        >>> from collections import Counter
        >>> freq = Counter({"chat": 3, "chien": 2, "oiseau": 1})
        >>> vocab, word2id, id2word = vocabulaire2(freq)
        >>> vocab  # ['chat', 'chien', 'oiseau']
        >>> word2id["chat"]  # 0
        >>> id2word[1]  # 'chien'
    """
    vocab = [w for w, c in frequence_mots.items()]
    vocab.sort()
    word2id = {}
    id2word = {}    
    for i in range(len(vocab)):
        assert vocab[i] not in word2id
        word2id[vocab[i]] = i
        id2word[i] = vocab[i]
    return vocab, word2id, id2word


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