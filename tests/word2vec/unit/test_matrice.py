
from src.word2vec.matrice import softmax
import numpy as np
import pytest

# On suppose la fonction:
# def softmax(matrice_centre, matrice_contexte, id_mot_centre):
#     v_c = matrice_centre[id_mot_centre]
#     s_j = np.dot(matrice_contexte, v_c)
#     exp_sj = np.exp(s_j - np.max(s_j))
#     return exp_sj / np.sum(exp_sj)

class TestSoftmax:
    """Unit tests for the efficient softmax implementation"""

    def test_basic_three_values(self):
        """Distribution correcte (trois valeurs) et somme = 1"""
        matrice_centre = np.array([
            [0.2, 0.8, 0.5],   # id 0
            [0.1, 0.3, 0.9],   # id 1
            [0.7, 0.2, 0.4]    # id 2
        ])
        matrice_contexte = np.array([
            [0.5, 0.1, 0.3],
            [0.9, 0.7, 0.2],
            [0.2, 0.4, 0.8]
        ])

        res = softmax(matrice_centre, matrice_contexte, 0)
        assert res.shape == (3,)
        assert np.all(res >= 0)
        assert np.isclose(np.sum(res), 1.0, atol=1e-8)

        # Vérif avec calcul "attendu"
        logits = matrice_contexte @ matrice_centre[0]
        expected = np.exp(logits - np.max(logits)) / np.sum(np.exp(logits - np.max(logits)))
        assert np.allclose(res, expected, atol=1e-8)

    def test_uniform_when_equal_logits(self):
        """Si tous les produits scalaires sont égaux → proba uniforme"""
        # même contexte, même v_c ⇒ mêmes logits
        matrice_centre = np.array([[1.0, 1.0], [2.0, 2.0]])
        matrice_contexte = np.array([[3.0, 4.0],
                                     [3.0, 4.0],
                                     [3.0, 4.0]])
        res = softmax(matrice_centre, matrice_contexte, 0)
        assert np.allclose(res, np.array([1/3, 1/3, 1/3]), atol=1e-8)

    def test_zero_center_vector(self):
        """v_c = 0 ⇒ logits tous nuls ⇒ proba uniforme"""
        matrice_centre = np.array([[0.0, 0.0]])
        matrice_contexte = np.random.rand(5, 2)
        res = softmax(matrice_centre, matrice_contexte, 0)
        assert np.allclose(res, np.full(5, 1/5), atol=1e-8)

    def test_numerical_stability_large_values(self):
        """Robustesse pour grandes valeurs (pas d'overflow)"""
        matrice_centre = np.full((2, 3), 1e3)
        matrice_contexte = np.full((4, 3), 1e3)
        res = softmax(matrice_centre, matrice_contexte, 1)
        # Les 4 logits sont égaux ⇒ uniforme
        assert np.allclose(res, np.full(4, 1/4), atol=1e-8)

    def test_shape_matches_num_context_rows(self):
        """La longueur de sortie = nb de lignes du contexte"""
        C, D = 7, 10
        matrice_centre = np.random.randn(C, D)
        matrice_contexte = np.random.randn(13, D)
        res = softmax(matrice_centre, matrice_contexte, 3)
        assert res.shape == (13,)

    def test_invalid_center_index_raises(self):
        """Index centre hors bornes ⇒ IndexError"""
        matrice_centre = np.random.randn(3, 5)
        matrice_contexte = np.random.randn(4, 5)
        with pytest.raises(IndexError):
            _ = softmax(matrice_centre, matrice_contexte, 10)

    def test_mismatched_dimensions_raises(self):
        """Dimensions incompatibles pour le produit scalaire ⇒ ValueError (NumPy)"""
        matrice_centre = np.random.randn(4, 8)   # dim 8
        matrice_contexte = np.random.randn(5, 6) # dim 6 (incompatible)
        with pytest.raises(ValueError):
            _ = softmax(matrice_centre, matrice_contexte, 2)

    def test_probabilities_change_with_center(self):
        """Changer le mot centre change la distribution (sauf cas dégénérés)"""
        matrice_centre = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ])
        matrice_contexte = np.array([
            [1.0, 0.0, 0.0],   # favorise centre 0
            [0.0, 1.0, 0.0],   # favorise centre 1
            [0.5, 0.5, 0.0]    # neutre entre 0 et 1
        ])
        p0 = softmax(matrice_centre, matrice_contexte, 0)
        p1 = softmax(matrice_centre, matrice_contexte, 1)
        assert not np.allclose(p0, p1)
        assert np.isclose(np.sum(p0), 1.0)
        assert np.isclose(np.sum(p1), 1.0)

    def test_returns_numpy_array(self):
        """Le type de retour est un ndarray (pas une liste)"""
        matrice_centre = np.random.randn(2, 3)
        matrice_contexte = np.random.randn(3, 3)
        res = softmax(matrice_centre, matrice_contexte, 1)
        assert isinstance(res, np.ndarray)
