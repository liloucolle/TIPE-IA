import torch
import numpy as np


from extraction.extraction import top_n_keywords_from_text
from content.echantillon_texte import article
from word2vec.matrice import deep_learning as softmax_numpy
from word2vec_negativesampling.word2vec_negativesampling import deep_learning_negative_sampling
from word2vec_pytorch.word2vec_pytorch import (
    SkipGram, SkipGramSoftmaxDS, SkipGramNegSampleDS,
    train_softmax, train_neg
)
from text_utils.text_utils import tokenisation, frequence_mots, vocabulaire, creation_paires


