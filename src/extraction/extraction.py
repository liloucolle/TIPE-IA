from content.echantillon_texte import (article)
from text_utils.text_utils import tokenisation, frequence_mots, vocabulaire
import numpy as np
import csv
import os
from datetime import datetime


def _load_english_stopwords():
    """Retourne un set de stop words anglais.
    Essaie plusieurs sources légères: scikit-learn, puis stop_words, sinon
    utilise une petite liste intégrée de secours.
    """
    try:
        from stop_words import get_stop_words
        return set(get_stop_words('en'))
    except Exception:
        pass

    # Petite liste de secours minimale
    return set([
        'a','an','the','and','or','in','on','at','to','for','from','by','with',
        'about','as','is','it','this','that','these','those','of','be','have',
        'has','had','are','was','were','but','if','then','so','out','up','down',
        'not'
    ])

def calcul_vecteur_moyen(embeddings, vocab, word2id):
    """
    Calcule le vecteur moyen des embeddings des mots donnés.
    """
    vectors = []
    for mot in vocab:
        e = embeddings[word2id[mot]]
        vectors.append(embeddings[word2id[mot]])
    if not vectors:
        return None
    return np.mean(vectors, axis=0)

def preprocess_text(text):
    """
    Prétraite le texte en le tokenisant et en calculant la fréquence des mots.
    Retourne le vocabulaire, word2id et id2word.
    """

    texte = tokenisation(text)
    frequence = frequence_mots(texte)

    # Filtrer les stop words anglais pour qu'ils n'apparaissent pas dans le vocabulaire
    stopwords = _load_english_stopwords()
    # frequence est un Counter; construire un nouveau Counter filtré
    filtered = {w: c for w, c in frequence.items() if w not in stopwords}
    vocab, word2id, id2word = vocabulaire(filtered)
    return vocab, word2id, id2word

def top_n_keywords_from_text(embeddings, text, N=10, csv_path=None, append=True, embeddings_type = "unknown"):
    """
    Retourne les top-N mots dont les embeddings ont la plus grande similarité cosinus
    avec le vecteur moyen des mots présents dans `text`.

    Params:
    - embeddings: dict mapping mot -> np.ndarray
    - text: chaîne de texte
    - n: nombre de mots à retourner (par défaut 10)

    Retourne une liste de tuples `(mot, similarite)` triée par similarité décroissante.
    """

    vocab, word2id, id2word = preprocess_text(text)
    len_vocab = len(vocab)

    mean_vec = calcul_vecteur_moyen(embeddings, vocab, word2id)
    if mean_vec is None:
        return []

    sims = []
    mean_norm = np.linalg.norm(mean_vec)
    if mean_norm == 0:
        return []

    for mot in vocab:
        vec = embeddings[word2id[mot]] # Assurer que c'est un tableau NumPy
        denominator = mean_norm * np.linalg.norm(vec)  # Normalisation, 'np.linalg.norm' calcule la norme euclidienne
        if denominator == 0:
            sim = 0.0
        else:
            sim = float(np.dot(mean_vec, vec) / denominator) # Similarité cosinus
        sims.append((mot, sim))

    sims.sort(key=lambda x: x[1], reverse=True) # Tri décroissant
    topk = sims[:N]

    # Si un chemin CSV est fourni, écrire/concaténer les résultats
    if csv_path:
        write_header = not (os.path.exists(csv_path) and os.path.getsize(csv_path) > 0)
        mode = 'a' if append else 'w'
        with open(csv_path, mode, newline='', encoding='utf-8') as f:
            fieldnames = ['timestamp', 'vocab_size', 'embeddings_type', 'rank', 'word', 'similarity']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            ts = datetime.utcnow().isoformat()
            for rank, (mot, sim) in enumerate(topk, start=1):
                writer.writerow({
                    'timestamp': ts,    
                    'vocab_size': len_vocab,
                    'embeddings_type': embeddings_type,
                    'rank': rank,
                    'word': mot,
                    'similarity': sim,
                })

    return topk
