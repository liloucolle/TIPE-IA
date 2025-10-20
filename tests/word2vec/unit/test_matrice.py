import pytest
import numpy as np
from src.word2vec.matrice import softmax





class TestSoftmax:
    """Unit tests for the softmax function"""

    def test_single_pair(self):
        """Test with a single pair of word vectors"""
        word2id = {"chat": 0, "chien": 1}
        matrice_centre = np.array([[0.2, 0.8], [0.6, 0.4]])
        matrice_contexte = np.array([[0.5, 0.1], [0.9, 0.7]])

        # Expected: softmax of the dot products for the chosen center word
        id_centre = word2id["chat"]
        logits = np.dot(matrice_contexte, matrice_centre[id_centre])
        expected = np.exp(logits) / np.sum(np.exp(logits))
        
        result = softmax(matrice_centre, matrice_contexte, id_centre, word2id)
        assert np.allclose(result, expected, atol=1e-6)

    def test_all_equal_vectors(self):
        """If all dot products are equal, the softmax should return uniform probabilities"""
        word2id = {"a": 0, "b": 1, "c": 2}
        matrice_centre = np.ones((3, 3))
        matrice_contexte = np.ones((3, 3))
        id_centre = word2id["a"]

        result = softmax(matrice_centre, matrice_contexte, id_centre, word2id)
        expected = np.array([1/3, 1/3, 1/3])
        assert np.allclose(result, expected, atol=1e-6)

    def test_high_magnitude_values(self):
        """Test numerical stability with large input values"""
        word2id = {"x": 0, "y": 1}
        matrice_centre = np.array([[1000, 1000], [1000, 1000]])
        matrice_contexte = np.array([[1000, 1000], [1000, 1000]])
        id_centre = 0

        result = softmax(matrice_centre, matrice_contexte, id_centre, word2id)
        expected = np.array([0.5, 0.5])
        assert np.allclose(result, expected, atol=1e-6)

    def test_negative_values(self):
        """Test that negative values are handled correctly"""
        word2id = {"x": 0, "y": 1}
        matrice_centre = np.array([[-1, -2], [3, 1]])
        matrice_contexte = np.array([[2, 1], [-3, -1]])
        id_centre = 0

        logits = np.dot(matrice_contexte, matrice_centre[id_centre])
        expected = np.exp(logits) / np.sum(np.exp(logits))
        result = softmax(matrice_centre, matrice_contexte, id_centre, word2id)
        assert np.allclose(result, expected, atol=1e-6)

    def test_output_shape(self):
        """Output should have the same length as the number of context words"""
        word2id = {"mot1": 0, "mot2": 1, "mot3": 2}
        matrice_centre = np.random.rand(3, 4)
        matrice_contexte = np.random.rand(3, 4)
        id_centre = 1

        result = softmax(matrice_centre, matrice_contexte, id_centre, word2id)
        assert result.shape == (3,)

    def test_invalid_id(self):
        """Test that function raises an error for invalid word id"""
        word2id = {"a": 0, "b": 1}
        matrice_centre = np.random.rand(2, 3)
        matrice_contexte = np.random.rand(2, 3)
        
        with pytest.raises(IndexError):
            softmax(matrice_centre, matrice_contexte, 5, word2id)

    def test_zero_vectors(self):
        """All-zero vectors should return uniform probabilities"""
        word2id = {"a": 0, "b": 1, "c": 2}
        matrice_centre = np.zeros((3, 3))
        matrice_contexte = np.zeros((3, 3))
        id_centre = 1

        result = softmax(matrice_centre, matrice_contexte, id_centre, word2id)
        expected = np.array([1/3, 1/3, 1/3])
        assert np.allclose(result, expected, atol=1e-6)

    def test_different_dimensions(self):
        """Check that mismatched matrix dimensions raise an error"""
        word2id = {"a": 0, "b": 1}
        matrice_centre = np.random.rand(2, 4)
        matrice_contexte = np.random.rand(2, 3)
        id_centre = 0

        with pytest.raises(ValueError):
            softmax(matrice_centre, matrice_contexte, id_centre, word2id)
