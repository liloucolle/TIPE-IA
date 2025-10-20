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