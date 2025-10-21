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
    """Unit tests for the frequence_mots function"""
    
    def test_simple_frequency(self):
        """Test frequency counting with simple text"""
        tokens = [["hello", "world"], ["hello", "test"]]
        result = frequence_mots(tokens)
        expected = Counter({"hello": 2, "world": 1, "test": 1})
        assert result == expected
    
    def test_empty_input(self):
        """Test with empty token list"""
        tokens = []
        result = frequence_mots(tokens)
        expected = Counter()
        assert result == expected
    
    def test_empty_sentences(self):
        """Test with empty sentences in token list"""
        tokens = [[], ["word"], []]
        result = frequence_mots(tokens)
        expected = Counter({"word": 1})
        assert result == expected
    
    def test_single_word_repeated(self):
        """Test with same word repeated multiple times"""
        tokens = [["cat", "cat"], ["cat", "dog"], ["cat"]]
        result = frequence_mots(tokens)
        expected = Counter({"cat": 4, "dog": 1})
        assert result == expected
    
    def test_no_repeated_words(self):
        """Test with all unique words"""
        tokens = [["apple", "banana"], ["cherry", "date"]]
        result = frequence_mots(tokens)
        expected = Counter({"apple": 1, "banana": 1, "cherry": 1, "date": 1})
        assert result == expected
    
    def test_single_sentence(self):
        """Test with single sentence"""
        tokens = [["the", "quick", "brown", "fox", "the"]]
        result = frequence_mots(tokens)
        expected = Counter({"the": 2, "quick": 1, "brown": 1, "fox": 1})
        assert result == expected
    
    def test_case_sensitive(self):
        """Test that function is case sensitive (should be since tokenisation handles case)"""
        tokens = [["Hello", "hello"], ["HELLO"]]
        result = frequence_mots(tokens)
        expected = Counter({"Hello": 1, "hello": 1, "HELLO": 1})
        assert result == expected


class TestVocabulaire2:
    """Unit tests for the vocabulaire2 function"""
    
    def test_simple_vocabulary(self):
        """Test vocabulary creation from simple word frequencies"""
        word_freq = Counter({"hello": 3, "world": 2, "test": 1})
        vocab, stoi, itos = vocabulaire(word_freq)
        
        # Check vocabulary is sorted
        expected_vocab = ["hello", "test", "world"]  # alphabetical order
        assert vocab == expected_vocab
        
        # Check stoi mapping
        expected_stoi = {"hello": 0, "test": 1, "world": 2}
        assert stoi == expected_stoi
        
        # Check itos mapping
        expected_itos = {0: "hello", 1: "test", 2: "world"}
        assert itos == expected_itos
    
    def test_empty_frequency(self):
        """Test with empty word frequency counter"""
        word_freq = Counter()
        vocab, stoi, itos = vocabulaire(word_freq)
        
        assert vocab == []
        assert stoi == {}
        assert itos == {}
    
    def test_single_word(self):
        """Test with single word in frequency counter"""
        word_freq = Counter({"word": 5})
        vocab, stoi, itos = vocabulaire(word_freq)
        
        assert vocab == ["word"]
        assert stoi == {"word": 0}
        assert itos == {0: "word"}
    
    def test_alphabetical_sorting(self):
        """Test that vocabulary is sorted alphabetically"""
        word_freq = Counter({"zebra": 1, "apple": 2, "banana": 3, "cherry": 1})
        vocab, stoi, itos = vocabulaire(word_freq)
        
        expected_vocab = ["apple", "banana", "cherry", "zebra"]
        assert vocab == expected_vocab
        
        # Verify indices match sorted order
        for i, word in enumerate(expected_vocab):
            assert stoi[word] == i
            assert itos[i] == word
    
    def test_consistency_stoi_itos(self):
        """Test that stoi and itos are inverse mappings"""
        word_freq = Counter({"cat": 4, "dog": 2, "bird": 1, "fish": 3})
        vocab, stoi, itos = vocabulaire(word_freq)
        
        # Check that stoi and itos are inverses
        for word in vocab:
            idx = stoi[word]
            assert itos[idx] == word
        
        for idx in itos:
            word = itos[idx]
            assert stoi[word] == idx
    
    def test_frequencies_ignored_in_sorting(self):
        """Test that word frequencies don't affect alphabetical sorting"""
        word_freq = Counter({"z": 100, "a": 1, "m": 50})
        vocab, stoi, itos = vocabulaire(word_freq)
        
        # Should be sorted alphabetically regardless of frequency
        expected_vocab = ["a", "m", "z"]
        assert vocab == expected_vocab
    
    def test_special_characters(self):
        """Test vocabulary with special characters in words"""
        word_freq = Counter({"word1": 1, "word_2": 2, "word-3": 1})
        vocab, stoi, itos = vocabulaire(word_freq)
        
        expected_vocab = ["word-3", "word1", "word_2"]  # ASCII order
        assert vocab == expected_vocab
        
        # Verify mappings
        assert len(stoi) == len(itos) == len(vocab)
        for i, word in enumerate(vocab):
            assert stoi[word] == i
            assert itos[i] == word

