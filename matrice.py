import numpy as np

# Choix de la dimmension de la matrice
dimension = 100
window_size = 2

# Exemple de texte
texte = 'Word embeddings translate words into vectors, revealing their hidden relationships within a continuous mathematical space.Thanks to them, machines can finally understand that “king” is to “queen” what “man” is to “woman.”'


# Tokenisation du texte

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
