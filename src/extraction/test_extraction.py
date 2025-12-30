import torch
import numpy as np


from content.echantillon_texte import *
from word2vec.matrice import deep_learning as softmax_numpy
from word2vec_negativesampling.word2vec_negativesampling import deep_learning_negative_sampling
from word2vec_pytorch.word2vec_pytorch import deep_learning_pytorch
from text_utils.text_utils import tokenisation, frequence_mots, vocabulaire, creation_paires
from extraction import top_n_keywords_from_text


texte = texte_100

embeddings = deep_learning_pytorch(texte)

top_keywords = top_n_keywords_from_text(embeddings, article, N=10)
print("Top 10 mots clés extraits :")
for mot, sim in top_keywords:
    print(f"{mot}: {sim:.4f}")
