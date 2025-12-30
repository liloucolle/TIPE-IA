import torch
import numpy as np


from content.echantillon_texte import *
from word2vec.matrice import deep_learning as softmax_numpy
from word2vec_negativesampling.word2vec_negativesampling import deep_learning_negative_sampling
from word2vec_pytorch.word2vec_pytorch import deep_learning_pytorch
from text_utils.text_utils import tokenisation, frequence_mots, vocabulaire, creation_paires
from extraction import top_n_keywords_from_text


texte = texte_10

embeddings = deep_learning_pytorch(texte)


csv_path = "keywords_results.csv"

top_keywords = top_n_keywords_from_text(embeddings, article, N=3, csv_path=csv_path, append=True)
print("Top 3 mots clés extraits :")
for mot, sim in top_keywords:
    print(f"{mot}: {sim:.4f}")
