import torch
import numpy as np


from content.echantillon_texte import texte_10,texte_100, article
from word2vec.matrice import deep_learning as softmax_numpy
from word2vec_negativesampling.word2vec_negativesampling import deep_learning_negative_sampling
from word2vec_pytorch.word2vec_pytorch import deep_learning_pytorch
from text_utils.text_utils import tokenisation, frequence_mots, vocabulaire, creation_paires
from extraction import top_n_keywords_from_text
import os
from datetime import datetime


def save_embeddings(embeddings, text_name, model_name, itos):
    """Save embeddings and index-to-string list to a compressed .npz file.
    embeddings: numpy array (V, D)
    itos: list mapping index -> token
    """
    from pathlib import Path
    import tempfile

    # Prefer an absolute output path to avoid relative-path issues
    out_dir = Path.cwd() / "output" / "embeddings"
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        # Fallback to the system temp directory if creation fails
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
    # Convert itos to numpy array of strings
    itos_arr = np.array(itos, dtype=object)
    np.savez_compressed(path, embeddings=embeddings, itos=itos_arr)
    print(f"[OK] Embeddings saved: {path}")

CORPORA = [
    #("Texte kings and Queen", texte_king_and_queen_2)
    ("Texte 10", texte_10),
    #("Texte 100", texte_100),
    #("Texte 1000", texte_1000),
    #("Texte 10000", texte_10000)
]


csv_path = "keywords_results.csv"

#==========================
# Paramètres communs
# =========================
params_numpy = dict(
    dimension=100,
    taux_apprentissage=0.1,
    window_size=2,
    n_epochs=100,
    patience=3,
    tolerance=1e-4,
    decay=0.95
)

article = article

# Params PyTorch (alignés sur la signature demandée)
params_torch_neg = dict(
    dimension=100,
    taux_apprentissage=13.1,
    window_size=2,
    n_epochs=100,
    mode="neg",          
    batch_size=256
)

params_torch_softmax = dict(
    dimension=100,
    taux_apprentissage=13.1,
    window_size=2,
    n_epochs=100,
    mode="softmax",          
    batch_size=256
)


for text_name, texte in CORPORA:
    print(f"\n=== Extraction des mots clés pour {text_name} ===")

    # Calcul des embeddings avec les différents modèles
    print("Calcul des embeddings avec Softmax (NumPy)...")
    embeddings = softmax_numpy(texte, **params_numpy)[0]
    # save embeddings (build vocab to get itos)
    freq = frequence_mots(tokenisation(texte))
    _, stoi, itos = vocabulaire(freq)
    save_embeddings(embeddings, text_name, "softmax_numpy", itos)

    top_keywords = top_n_keywords_from_text(embeddings, article, N=3, csv_path=csv_path, append=True, embeddings_type="Softmax NumPy")
    print("Top 3 mots clés extraits :")
    for mot, sim in top_keywords:
        print(f"{mot}: {sim:.4f}")

    print("Calcul des embeddings avec Negative Sampling (NumPy)...")
    embeddings = deep_learning_negative_sampling(texte, **params_numpy)[0]
    freq = frequence_mots(tokenisation(texte))
    _, stoi, itos = vocabulaire(freq)
    save_embeddings(embeddings, text_name, "neg_numpy", itos)

    top_keywords = top_n_keywords_from_text(embeddings, article, N=3, csv_path=csv_path, append=True, embeddings_type="NegSampling NumPy")
    print("Top 3 mots clés extraits :")
    for mot, sim in top_keywords:
        print(f"{mot}: {sim:.4f}")

    print("Calcul des embeddings avec Softmax (PyTorch)...")
    embeddings = deep_learning_pytorch(texte, **params_torch_softmax)
    freq = frequence_mots(tokenisation(texte))
    _, stoi, itos = vocabulaire(freq)
    save_embeddings(embeddings, text_name, "pt_softmax", itos)

    top_keywords = top_n_keywords_from_text(embeddings, article, N=3, csv_path=csv_path, append=True, embeddings_type="PyTorch Softmax")
    print("Top 3 mots clés extraits :")
    for mot, sim in top_keywords:
        print(f"{mot}: {sim:.4f}")

    print("Calcul des embeddings avec Negative Sampling (PyTorch)...")
    embeddings = deep_learning_pytorch(texte, **params_torch_neg)
    freq = frequence_mots(tokenisation(texte))
    _, stoi, itos = vocabulaire(freq)
    save_embeddings(embeddings, text_name, "pt_neg", itos)

    top_keywords = top_n_keywords_from_text(embeddings, article, N=3, csv_path=csv_path, append=True, embeddings_type="PyTorch Negative Sampling")
    print("Top 3 mots clés extraits :")
    for mot, sim in top_keywords:
        print(f"{mot}: {sim:.4f}")
