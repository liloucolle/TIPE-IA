import numpy as np

# Choix de la dimmension de la matrice
dimension = 100

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
    
    return mots


    

texte = 'Word embeddings translate words into vectors, revealing their hidden relationships within a continuous mathematical space.Thanks to them, machines can finally understand that “king” is to “queen” what “man” is to “woman.”'


def vocabulaire (texte):

    t = tokenisation(texte)
    vocab = {}
    for i in range(len(t)):
        vocab[i] = t[i]
    
    return vocab

print(vocabulaire(texte))

def intialisation_matrice (vocab):

    n = len(vocab)
    matrice_centre = np.random.rand(n,dimension)
    matrice_contexte = np.random.rand(n,dimension)

    return matrice_centre, matrice_contexte

