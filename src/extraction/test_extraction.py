"""
Script de test d'extraction de mots-clés avec évaluation sémantique.

Workflow:
1. Entraîner 4 modèles d'embeddings (2 NumPy + 2 PyTorch)
2. Extraire les mots-clés des textes principaux
3. Évaluer la qualité sémantique des mots-clés sur les 10 articles
4. Sauvegarder les scores dans keyword_scores_test.csv
"""
import torch
import numpy as np

from content.echantillon_texte import texte_10, texte_100, texte_1000, texte_10000, articles, mots_representatifs, article
from word2vec.word2vec import deep_learning as softmax_numpy
from word2vec_negativesampling.word2vec_negativesampling import deep_learning_negative_sampling
from word2vec_pytorch.word2vec_pytorch import deep_learning_pytorch
from text_utils.text_utils import tokenisation, frequence_mots, vocabulaire
from extraction import top_n_keywords_from_text, score_semantique, preprocess_text
import os
from datetime import datetime



def save_embeddings(embeddings, text_name, model_name, itos):
    """
    Sauvegarder les embeddings et le vocabulaire dans un fichier .npz compressé.

    Args:
        embeddings: matrice numpy (V, D) contenant les embeddings
        text_name: nom du texte d'entraînement
        model_name: nom du modèle
        itos: liste mappant index -> token (vocabulaire inverse)
    """
    from pathlib import Path
    import tempfile

    # Préfère un chemin de sortie absolu pour éviter les problèmes de chemins relatifs
    out_dir = Path.cwd() / "output" / "embeddings"
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        # Repli sur le dossier temporaire système si la création échoue
        fallback = Path(tempfile.gettempdir()) / "embeddings"
        try:
            fallback.mkdir(parents=True, exist_ok=True)
            out_dir = fallback
            print(f"[WARN] Could not create {out_dir}; falling back to {fallback}: {e}")
        except Exception:
            raise

    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    safe_text = text_name.replace(" ", "_")
    safe_model = model_name.replace(" ", "_")
    fname = f"embeddings_{safe_model}_{safe_text}_{stamp}.npz"
    path = out_dir / fname
    # Convertir itos en tableau NumPy de chaînes
    itos_arr = np.array(itos, dtype=object)
    np.savez_compressed(path, embeddings=embeddings, itos=itos_arr)
    print(f"[OK] Embeddings saved: {path}")


def evaluate_model_on_articles(embeddings, model_type, csv_scores_path, n_epochs, stoi):
    """
    Évaluer un modèle entraîné sur tous les 10 articles et sauvegarder les scores.

    Pour chaque article:
    1. Extraire les top 5 mots-clés avec ce modèle
    2. Comparer avec les mots attendus (ground truth)
    3. Calculer un score de validité sémantique (0-100%)
    4. Sauvegarder tous les résultats dans un CSV

    Args:
        embeddings: matrice d'embeddings du modèle
        model_type: nom du modèle (ex: "PyTorch Softmax")
        csv_scores_path: chemin du fichier CSV de sortie
        n_epochs: nombre d'epochs d'entraînement
        stoi: vocabulaire string-to-index
    """
    import csv

    # Initialiser le fichier CSV s'il n'existe pas
    file_exists = os.path.isfile(csv_scores_path)

    with open(csv_scores_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Écrire l'en-tête au premier passage
        if not file_exists:
            writer.writerow([
                'timestamp', 'model_type', 'article_id', 'validity_score',
                'extracted_keywords', 'expected_keywords', 'vocab_size', 'n_epochs'
            ])

        # Évaluer le modèle sur chaque article
        for article_id, article_text in enumerate(articles, start=1):
            expected_keywords = mots_representatifs[article_id]

            # Extraire les top 5 mots-clés du modèle
            extracted_kw_tuples = top_n_keywords_from_text(
                embeddings, article_text, N=5, csv_path=None, embeddings_type=model_type
            )
            extracted_keywords = [kw for kw, _ in extracted_kw_tuples]

            # Obtenir le vocabulaire et le mapping mot->index pour calculer les scores
            _, word2id, _ = preprocess_text(article_text)

            # Calculer le score de validité sémantique (0-100%)
            validity_score = score_semantique(
                extracted_keywords, expected_keywords, embeddings, word2id
            )

            # Enregistrer le résultat
            timestamp = datetime.utcnow().isoformat()
            vocab_size = len(stoi)

            writer.writerow([
                timestamp,
                model_type,
                article_id,
                f"{validity_score:.2f}",
                ",".join(extracted_keywords),
                ",".join(expected_keywords),
                vocab_size,
                n_epochs
            ])

    print(f"[OK] Résultats d'évaluation sauvegardés: {csv_scores_path}")


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Corpus de textes à traiter (textes principaux pour l'entraînement des modèles)
CORPORA = [
    #("Texte 10", texte_10),
    ("Texte 100", texte_100),
    #("Texte 1000", texte_1000),
    #("Texte 10000", texte_10000)



]

# Répertoire de sortie pour les résultats
os.makedirs("figures", exist_ok=True)
csv_path = "figures/keywords_results.csv"  # Top 3 mots-clés extraits
csv_scores_path = "figures/keyword_scores_test.csv"  # Scores d'évaluation sémantique

# Paramètres d'entraînement pour les modèles NumPy
params_numpy = dict(
    dimension=100,
    taux_apprentissage=0.1,
    window_size=2,
    n_epochs=100,
    decay=0.95
)

# Paramètres d'entraînement pour les modèles PyTorch (Negative Sampling)
params_torch_neg = dict(
    dimension=100,
    taux_apprentissage=13.1,
    window_size=2,
    n_epochs=100,
    mode="neg",
    batch_size=256
)

# Paramètres d'entraînement pour les modèles PyTorch (Softmax)
params_torch_softmax = dict(
    dimension=100,
    taux_apprentissage=13.1,
    window_size=2,
    n_epochs=100,
    mode="softmax",
    batch_size=256
)


# ==============================================================================
# ENTRAÎNEMENT ET ÉVALUATION DES 4 MODÈLES
# ==============================================================================
for text_name, texte in CORPORA:
    print(f"\n=== Traitement du corpus: {text_name} ===")

    # ---------- MODÈLE 1: Softmax NumPy ----------
    print("\n1) Entraînement du modèle Softmax NumPy...")
    embeddings = softmax_numpy(texte, **params_numpy)[0]
    freq = frequence_mots(tokenisation(texte))
    _, stoi, itos = vocabulaire(freq)
    save_embeddings(embeddings, text_name, "softmax_numpy", itos)

    # Extraire top 3 mots-clés du texte principal
    top_keywords = top_n_keywords_from_text(embeddings, article, N=3, csv_path=csv_path, append=True, embeddings_type="Softmax NumPy")
    print("   Top 3 mots-clés extraits:")
    for mot, sim in top_keywords:
        print(f"   - {mot}: {sim:.4f}")

    # Évaluer sur tous les 10 articles
    print("   Évaluation sur les 10 articles...")
    evaluate_model_on_articles(embeddings, "Softmax NumPy", csv_scores_path, params_numpy["n_epochs"], stoi)

    # ---------- MODÈLE 2: Negative Sampling NumPy ----------
    print("\n2) Entraînement du modèle Negative Sampling NumPy...")
    embeddings = deep_learning_negative_sampling(texte, **params_numpy)[0]
    freq = frequence_mots(tokenisation(texte))
    _, stoi, itos = vocabulaire(freq)
    save_embeddings(embeddings, text_name, "neg_numpy", itos)

    # Extraire top 3 mots-clés du texte principal
    top_keywords = top_n_keywords_from_text(embeddings, article, N=3, csv_path=csv_path, append=True, embeddings_type="NegSampling NumPy")
    print("   Top 3 mots-clés extraits:")
    for mot, sim in top_keywords:
        print(f"   - {mot}: {sim:.4f}")

    # Évaluer sur tous les 10 articles
    print("   Évaluation sur les 10 articles...")
    evaluate_model_on_articles(embeddings, "NegSampling NumPy", csv_scores_path, params_numpy["n_epochs"], stoi)

    # ---------- MODÈLE 3: PyTorch Softmax ----------
    print("\n3) Entraînement du modèle PyTorch Softmax...")
    embeddings = deep_learning_pytorch(texte, **params_torch_softmax)
    freq = frequence_mots(tokenisation(texte))
    _, stoi, itos = vocabulaire(freq)
    save_embeddings(embeddings, text_name, "pt_softmax", itos)

    # Extraire top 3 mots-clés du texte principal
    top_keywords = top_n_keywords_from_text(embeddings, article, N=3, csv_path=csv_path, append=True, embeddings_type="PyTorch Softmax")
    print("   Top 3 mots-clés extraits:")
    for mot, sim in top_keywords:
        print(f"   - {mot}: {sim:.4f}")

    # Évaluer sur tous les 10 articles
    print("   Évaluation sur les 10 articles...")
    evaluate_model_on_articles(embeddings, "PyTorch Softmax", csv_scores_path, params_torch_softmax["n_epochs"], stoi)

    # ---------- MODÈLE 4: PyTorch Negative Sampling ----------
    print("\n4) Entraînement du modèle PyTorch Negative Sampling...")
    embeddings = deep_learning_pytorch(texte, **params_torch_neg)
    freq = frequence_mots(tokenisation(texte))
    _, stoi, itos = vocabulaire(freq)
    save_embeddings(embeddings, text_name, "pt_neg", itos)

    # Extraire top 3 mots-clés du texte principal
    top_keywords = top_n_keywords_from_text(embeddings, article, N=3, csv_path=csv_path, append=True, embeddings_type="PyTorch Negative Sampling")
    print("   Top 3 mots-clés extraits:")
    for mot, sim in top_keywords:
        print(f"   - {mot}: {sim:.4f}")

    # Évaluer sur tous les 10 articles
    print("   Évaluation sur les 10 articles...")
    evaluate_model_on_articles(embeddings, "PyTorch Negative Sampling", csv_scores_path, params_torch_neg["n_epochs"], stoi)

print("\n=== Traitement terminé ===")
print(f"Résultats sauvegardés dans:")
print(f"  - {csv_path} (mots-clés extraits)")
print(f"  - {csv_scores_path} (scores d'évaluation sémantique)")

