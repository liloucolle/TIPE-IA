from content.echantillon_texte import article, articles, mots_representatifs
from text_utils.text_utils import tokenisation, frequence_mots, vocabulaire
from word2vec.word2vec import deep_learning as softmax_numpy
from word2vec_negativesampling.word2vec_negativesampling import deep_learning_negative_sampling
from word2vec_pytorch.word2vec_pytorch import deep_learning_pytorch
import numpy as np
import csv
import os
from datetime import datetime


def _load_english_stopwords():
    """Charger les stop words anglais (mots très communs à exclure de l'analyse)."""
    try:
        from nltk.corpus import stopwords
        return set(stopwords.words('english'))
    except Exception:
        pass
    # Liste de secours si NLTK n'est pas disponible
    return set([
        'a','an','the','and','or','in','on','at','to','for','from','by','with',
        'about','as','is','it','this','that','these','those','of','be','have',
        'has','had','are','was','were','but','if','then','so','out','up','down',
        'not'
    ])


def calcul_vecteur_moyen(embeddings, vocab, word2id):
    """Calculer le vecteur moyen des embeddings d'un vocabulaire."""
    vectors = [embeddings[word2id[mot]] for mot in vocab]
    if not vectors:
        return None
    return np.mean(vectors, axis=0)


def preprocess_text(text):
    """Prétraiter un texte: tokenisation, filtrage des stop words, création vocabulaire."""
    texte = tokenisation(text)
    frequence = frequence_mots(texte)
    stopwords = _load_english_stopwords()
    # Filtrer les stop words du vocabulaire
    filtered = {w: c for w, c in frequence.items() if w not in stopwords}
    vocab, word2id, id2word = vocabulaire(filtered)
    return vocab, word2id, id2word


def top_n_keywords_from_text(embeddings, text, N=10, csv_path=None, append=True, embeddings_type="unknown"):
    """
    Extraire les N mots-clés ayant la plus grande similarité cosinus avec le vecteur moyen du texte.

    Retourne: liste de tuples (mot, similarité) triée par similarité décroissante.
    Si csv_path est fourni, sauvegarde les résultats dans un CSV.
    """
    vocab, word2id, id2word = preprocess_text(text)
    len_vocab = len(vocab)

    mean_vec = calcul_vecteur_moyen(embeddings, vocab, word2id)
    if mean_vec is None:
        return []

    # Calculer similarité cosinus pour chaque mot du vocabulaire
    sims = []
    mean_norm = np.linalg.norm(mean_vec)
    if mean_norm == 0:
        return []

    for mot in vocab:
        vec = embeddings[word2id[mot]]
        denominator = mean_norm * np.linalg.norm(vec)
        sim = float(np.dot(mean_vec, vec) / denominator) if denominator != 0 else 0.0
        sims.append((mot, sim))

    sims.sort(key=lambda x: x[1], reverse=True)
    topk = sims[:N]

    # Optionnel: sauvegarder les résultats dans un CSV
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



def cosinus(u, v):
    """Calculer la similarité cosinus entre deux vecteurs u et v."""
    denominateur = np.linalg.norm(u) * np.linalg.norm(v)
    return float(np.dot(u, v) / denominateur) if denominateur != 0 else 0.0


def score_semantique(mots_extraits, mots_attendus, embeddings, word2id):
    """
    Évaluer la qualité sémantique des mots extraits par rapport aux mots attendus.

    Pour chaque mot extrait, on calcule sa meilleure similarité avec les mots attendus.
    Retourne un score moyen entre 0 et 100 (pourcentage).
    """
    scores = []

    for mot_extrait in mots_extraits:
        if mot_extrait not in word2id:
            continue

        vec_extrait = embeddings[word2id[mot_extrait]]
        meilleurs_scores = []

        for mot_attendu in mots_attendus:
            if mot_attendu not in word2id:
                continue
            vec_attendu = embeddings[word2id[mot_attendu]]
            meilleurs_scores.append(cosinus(vec_extrait, vec_attendu))

        if meilleurs_scores:
            scores.append(max(meilleurs_scores))

    if not scores:
        return 0.0

    score = sum(scores) / len(scores)
    return 100 * max(0.0, score)



def evaluer_mots_cles(
    embeddings,
    text,
    mots_attendus,
    N=5,
    csv_path="figures/pourcentage_similarite.csv",
    embeddings_type="unknown",
    nom_article="unknown"
):
    """
    Évaluer les mots-clés extraits d'un texte contre les mots attendus.
    Retourne le score et la liste des mots extraits.
    Optionnellement, sauvegarde le résultat dans un CSV.
    """
    # Extraire les N mots-clés
    mots_extraits_scores = top_n_keywords_from_text(
        embeddings=embeddings,
        text=text,
        N=N,
        csv_path=None,
        embeddings_type=embeddings_type
    )
    mots_extraits = [mot for mot, sim in mots_extraits_scores]

    # Calculer le score sémantique
    vocab, word2id, _ = preprocess_text(text)
    score = score_semantique(
        mots_extraits=mots_extraits,
        mots_attendus=mots_attendus,
        embeddings=embeddings,
        word2id=word2id
    )

    # Sauvegarder optionnellement dans un CSV
    if csv_path:
        os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)
        write_header = not (os.path.exists(csv_path) and os.path.getsize(csv_path) > 0)

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            fieldnames = [
                "timestamp",
                "article",
                "embeddings_type",
                "mots_extraits",
                "mots_attendus",
                "score_pourcentage"
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            if write_header:
                writer.writeheader()

            writer.writerow({
                "timestamp": datetime.utcnow().isoformat(),
                "article": nom_article,
                "embeddings_type": embeddings_type,
                "mots_extraits": ", ".join(mots_extraits),
                "mots_attendus": ", ".join(mots_attendus),
                "score_pourcentage": round(score, 2)
            })

    return score, mots_extraits


def evaluer_mots_cles_pour_corpora(
    corpus_texts=None,
    mots_attendus_dict=None,
    csv_path="figures/keyword_scores.csv",
    N=5,
    params_numpy=None,
    params_torch=None,
    append=False,
    verbose=True
):
    """
    Entraîner tous les modèles d'embeddings et évaluer les mots-clés pour chaque texte.

    Pour chaque texte du corpus, entraîne 4 modèles (2 NumPy + 2 PyTorch), extrait
    les mots-clés, compare avec les mots attendus, et sauvegarde les scores.

    Args:
        corpus_texts: liste de textes (par défaut: articles importés)
        mots_attendus_dict: dictionnaire {index: [mots_attendus]} (par défaut: mots_representatifs)
        csv_path: chemin du CSV de sortie
        N: nombre de mots-clés à extraire
        params_numpy: paramètres pour les modèles NumPy
        params_torch: paramètres pour les modèles PyTorch
        append: si True, ajoute au CSV; sinon écrase
        verbose: affiche la progression

    Retourne: liste de dictionnaires avec les résultats
    """
    # Utiliser les valeurs par défaut si non fournies
    if corpus_texts is None:
        corpus_texts = articles
    if mots_attendus_dict is None:
        mots_attendus_dict = mots_representatifs

    # Paramètres NumPy par défaut
    if params_numpy is None:
        params_numpy = dict(
            dimension=100,
            taux_apprentissage=0.1,
            window_size=2,
            n_epochs=100,
            decay=0.95
        )

    # Paramètres PyTorch par défaut
    if params_torch is None:
        params_torch = dict(
            dimension=100,
            taux_apprentissage=13.1,
            window_size=2,
            n_epochs=100,
            batch_size=256
        )

    # Préparer le fichier CSV
    if not append and os.path.exists(csv_path):
        os.remove(csv_path)
    os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)

    # Définir les 4 modèles à entraîner et évaluer
    model_builders = [
        ("Softmax NumPy", lambda text: softmax_numpy(text, **params_numpy)[0]),
        ("NegSampling NumPy", lambda text: deep_learning_negative_sampling(text, **params_numpy)[0]),
        ("PyTorch Softmax", lambda text: deep_learning_pytorch(text, mode="softmax", **{k: v for k, v in params_torch.items() if k != 'mode'})),
        ("PyTorch Negative Sampling", lambda text: deep_learning_pytorch(text, mode="neg", **{k: v for k, v in params_torch.items() if k != 'mode'})),
    ]

    results = []

    # Pour chaque texte du corpus
    for index, text in enumerate(corpus_texts, start=1):
        mots_attendus = mots_attendus_dict.get(index)
        if mots_attendus is None:
            if verbose:
                print(f"[WARN] Aucun mot attendu pour l'article {index}")
            continue

        article_name = f"article_{index}"

        # Entraîner et évaluer chaque modèle
        for embeddings_type, build_embeddings in model_builders:
            if verbose:
                print(f"Évaluation article {index} ({article_name}) avec {embeddings_type}...")
            try:
                embeddings = build_embeddings(text)
            except Exception as exc:
                if verbose:
                    print(f"[ERROR] Modèle {embeddings_type} échoué pour article {index}: {exc}")
                continue

            score, mots_extraits = evaluer_mots_cles(
                embeddings=embeddings,
                text=text,
                mots_attendus=mots_attendus,
                N=N,
                csv_path=csv_path,
                embeddings_type=embeddings_type,
                nom_article=article_name
            )

            results.append({
                "article_index": index,
                "article_name": article_name,
                "embeddings_type": embeddings_type,
                "score_pourcentage": round(score, 2),
                "mots_extraits": mots_extraits,
                "mots_attendus": mots_attendus,
            })

    return results