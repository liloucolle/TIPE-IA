import pytest
import numpy as np
from collections import Counter
from text_utils.text_utils import tokenisation, vocabulaire, frequence_mots


class TestTokenisationPhrases:
    def test_basic_two_sentences(self):
        texte = ["The King loves the Queen.", "The queen rules the kingdom!"]
        expected = [
            ["the", "king", "loves", "the", "queen"],
            ["the", "queen", "rules", "the", "kingdom"],
        ]
        assert tokenisation(texte) == expected

    def test_mixed_case_lowercasing(self):
        texte = ["HELLO World", "PyThOn Is FUN"]
        expected = [
            ["hello", "world"],
            ["python", "is", "fun"],
        ]
        assert tokenisation(texte) == expected

    def test_punctuation_and_numbers(self):
        texte = ["hello, world! 123go", "v2 is here: wow!!!"]
        expected = [
            ["hello", "world", "go"],
            ["v", "is", "here", "wow"],  # '2' et ':' sont des séparateurs
        ]
        assert tokenisation(texte) == expected

    def test_accents_are_separators(self):
        # accents (éàçô…) ne sont pas dans a-z -> considérés comme séparateurs
        texte = ["Café crème, déjà vu"]
        expected = [
            ["caf", "cr", "me", "d", "j", "vu"]
        ]
        assert tokenisation(texte) == expected

    def test_whitespace_variants(self):
        texte = ["hello\tworld", "line1\nline2\r\nend"]
        expected = [
            ["hello", "world"],
            ["line", "line", "end"],
        ]
        assert tokenisation(texte) == expected

    def test_empty_sentence_kept_as_empty_list(self):
        # Ta version actuelle ajoute la phrase même si elle est vide -> []
        texte = ["Hello world!", ""]
        expected = [
            ["hello", "world"],
            []
        ]
        assert tokenisation(texte) == expected

    def test_only_separators_sentence(self):
        texte = ["123!@#", "??--", "keep,going"]
        expected = [
            [],        # rien d'alphabétique
            [],        # idem
            ["keep", "going"],
        ]
        assert tokenisation(texte) == expected

    def test_word_at_end_captured(self):
        texte = ["ends with word", "no punctuation here"]
        expected = [
            ["ends", "with", "word"],
            ["no", "punctuation", "here"],
        ]
        assert tokenisation(texte) == expected

    def test_single_letter_words(self):
        texte = ["A b C", "x y z"]
        expected = [
            ["a", "b", "c"],
            ["x", "y", "z"],
        ]
        assert tokenisation(texte) == expected

    def test_return_is_list_of_lists(self):
        texte = ["Hello world"]
        out = tokenisation(texte)
        assert isinstance(out, list)
        assert all(isinstance(s, list) for s in out)
        assert all(all(isinstance(w, str) for w in s) for s in out)

class TestFrequenceMots:
    """Tests unitaires pour la fonction frequence_mots"""

    def test_simple_frequency(self):
        """Vérifie le comptage des fréquences pour un texte simple"""
        tokens = [["hello", "world"], ["hello", "test"]]
        result = frequence_mots(tokens)
        expected = Counter({"hello": 2, "world": 1, "test": 1})
        assert result == expected
    
    def test_empty_input(self):
        """Test avec une liste de tokens vide"""
        tokens = []
        result = frequence_mots(tokens)
        expected = Counter()
        assert result == expected
    
    def test_empty_sentences(self):
        """Test avec des phrases vides dans la liste de tokens"""
        tokens = [[], ["word"], []]
        result = frequence_mots(tokens)
        expected = Counter({"word": 1})
        assert result == expected
    
    def test_single_word_repeated(self):
        """Test avec un même mot répété plusieurs fois"""
        tokens = [["cat", "cat"], ["cat", "dog"], ["cat"]]
        result = frequence_mots(tokens)
        expected = Counter({"cat": 4, "dog": 1})
        assert result == expected
    
    def test_no_repeated_words(self):
        """Test avec tous les mots uniques"""
        tokens = [["apple", "banana"], ["cherry", "date"]]
        result = frequence_mots(tokens)
        expected = Counter({"apple": 1, "banana": 1, "cherry": 1, "date": 1})
        assert result == expected
    
    def test_single_sentence(self):
        """Test avec une seule phrase"""
        tokens = [["the", "quick", "brown", "fox", "the"]]
        result = frequence_mots(tokens)
        expected = Counter({"the": 2, "quick": 1, "brown": 1, "fox": 1})
        assert result == expected
    
    def test_case_sensitive(self):
        """Vérifie la sensibilité à la casse (la tokenisation conserve la casse)"""
        tokens = [["Hello", "hello"], ["HELLO"]]
        result = frequence_mots(tokens)
        expected = Counter({"Hello": 1, "hello": 1, "HELLO": 1})
        assert result == expected


class TestVocabulaire2:
    """Tests unitaires pour la fonction vocabulaire2"""
    
    def test_simple_vocabulary(self):
        """Vérifie la création du vocabulaire à partir de fréquences simples"""
        word_freq = Counter({"hello": 3, "world": 2, "test": 1})
        vocab, stoi, itos = vocabulaire(word_freq)
        
        # Vérifie que le vocabulaire est trié
        expected_vocab = ["hello", "test", "world"]  # ordre alphabétique
        assert vocab == expected_vocab
        
        # Vérifie le mapping stoi
        expected_stoi = {"hello": 0, "test": 1, "world": 2}
        assert stoi == expected_stoi
        
        # Vérifie le mapping itos
        expected_itos = {0: "hello", 1: "test", 2: "world"}
        assert itos == expected_itos
    
    def test_empty_frequency(self):
        """Test avec un compteur de fréquences de mots vide"""
        word_freq = Counter()
        vocab, stoi, itos = vocabulaire(word_freq)
        
        assert vocab == []
        assert stoi == {}
        assert itos == {}
    
    def test_single_word(self):
        """Test avec un seul mot dans le compteur de fréquences"""
        word_freq = Counter({"word": 5})
        vocab, stoi, itos = vocabulaire(word_freq)
        
        assert vocab == ["word"]
        assert stoi == {"word": 0}
        assert itos == {0: "word"}
    
    def test_alphabetical_sorting(self):
        """Vérifie que le vocabulaire est trié alphabétiquement"""
        word_freq = Counter({"zebra": 1, "apple": 2, "banana": 3, "cherry": 1})
        vocab, stoi, itos = vocabulaire(word_freq)
        
        expected_vocab = ["apple", "banana", "cherry", "zebra"]
        assert vocab == expected_vocab
        
        # Vérifie que les indices correspondent à l'ordre trié
        for i, word in enumerate(expected_vocab):
            assert stoi[word] == i
            assert itos[i] == word
    
    def test_consistency_stoi_itos(self):
        """Vérifie que stoi et itos sont des mappings inverses"""
        word_freq = Counter({"cat": 4, "dog": 2, "bird": 1, "fish": 3})
        vocab, stoi, itos = vocabulaire(word_freq)
        
        # Vérifie que stoi et itos sont inverses
        for word in vocab:
            idx = stoi[word]
            assert itos[idx] == word
        
        for idx in itos:
            word = itos[idx]
            assert stoi[word] == idx
    
    def test_frequencies_ignored_in_sorting(self):
        """Vérifie que les fréquences n'affectent pas le tri alphabétique"""
        word_freq = Counter({"z": 100, "a": 1, "m": 50})
        vocab, stoi, itos = vocabulaire(word_freq)
        
        # Doit être trié alphabétiquement indépendamment des fréquences
        expected_vocab = ["a", "m", "z"]
        assert vocab == expected_vocab
    
    def test_special_characters(self):
        """Vérifie le vocabulaire avec des caractères spéciaux dans les mots"""
        word_freq = Counter({"word1": 1, "word_2": 2, "word-3": 1})
        vocab, stoi, itos = vocabulaire(word_freq)
        
        expected_vocab = ["word-3", "word1", "word_2"]  # ordre ASCII
        assert vocab == expected_vocab
        
        # Vérifie les mappings
        assert len(stoi) == len(itos) == len(vocab)
        for i, word in enumerate(vocab):
            assert stoi[word] == i
            assert itos[i] == word

