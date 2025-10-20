import pytest
import numpy as np
from collections import Counter
from src.text_utils.text_utils import tokenisation, vocabulaire2, frequence_mots


class TestTokenisation:
    """Unit tests for the tokenisation function using pytest"""
    
    def test_simple_text(self):
        """Test tokenization of simple text"""
        text = "Hello world"
        expected = ["hello", "world"]
        result = tokenisation(text)
        assert result == expected
    
    def test_mixed_case(self):
        """Test that uppercase letters are converted to lowercase"""
        text = "Hello WORLD"
        expected = ["hello", "world"]
        result = tokenisation(text)
        assert result == expected
    
    def test_punctuation_removal(self):
        """Test that punctuation is removed and words are split correctly"""
        text = "Hello, world! How are you?"
        expected = ["hello", "world", "how", "are", "you"]
        result = tokenisation(text)
        assert result == expected
    
    def test_numbers_removed(self):
        """Test that numbers are treated as separators"""
        text = "Hello123world456test"
        expected = ["hello", "world", "test"]
        result = tokenisation(text)
        assert result == expected
    
    def test_multiple_separators(self):
        """Test handling of multiple consecutive non-alphabetic characters"""
        text = "Hello!!!world???test"
        expected = ["hello", "world", "test"]
        result = tokenisation(text)
        assert result == expected
    
    def test_empty_string(self):
        """Test tokenization of empty string"""
        text = ""
        expected = []
        result = tokenisation(text)
        assert result == expected
    
    def test_only_separators(self):
        """Test string with only non-alphabetic characters"""
        text = "123!@#$%^&*()"
        expected = []
        result = tokenisation(text)
        assert result == expected
    
    def test_single_word(self):
        """Test tokenization of single word"""
        text = "hello"
        expected = ["hello"]
        result = tokenisation(text)
        assert result == expected
    
    def test_word_at_end(self):
        """Test that word at end of string is captured"""
        text = "hello world"
        expected = ["hello", "world"]
        result = tokenisation(text)
        assert result == expected
    
    def test_french_text(self):
        """Test with French text (accented characters should be treated as separators)"""
        text = "Bonjour, ceci est un test avec des caractères spéciaux: é, à, ç"
        expected = ["bonjour", "ceci", "est", "un", "test", "avec", "des", "caract", "res", "sp", "ciaux"]
        result = tokenisation(text)
        assert result == expected
    
    def test_whitespace_handling(self):
        """Test various whitespace characters"""
        text = "hello\tworld\ntest\r\nend"
        expected = ["hello", "world", "test", "end"]
        result = tokenisation(text)
        assert result == expected


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
        vocab, stoi, itos = vocabulaire2(word_freq)
        
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
        vocab, stoi, itos = vocabulaire2(word_freq)
        
        assert vocab == []
        assert stoi == {}
        assert itos == {}
    
    def test_single_word(self):
        """Test with single word in frequency counter"""
        word_freq = Counter({"word": 5})
        vocab, stoi, itos = vocabulaire2(word_freq)
        
        assert vocab == ["word"]
        assert stoi == {"word": 0}
        assert itos == {0: "word"}
    
    def test_alphabetical_sorting(self):
        """Test that vocabulary is sorted alphabetically"""
        word_freq = Counter({"zebra": 1, "apple": 2, "banana": 3, "cherry": 1})
        vocab, stoi, itos = vocabulaire2(word_freq)
        
        expected_vocab = ["apple", "banana", "cherry", "zebra"]
        assert vocab == expected_vocab
        
        # Verify indices match sorted order
        for i, word in enumerate(expected_vocab):
            assert stoi[word] == i
            assert itos[i] == word
    
    def test_consistency_stoi_itos(self):
        """Test that stoi and itos are inverse mappings"""
        word_freq = Counter({"cat": 4, "dog": 2, "bird": 1, "fish": 3})
        vocab, stoi, itos = vocabulaire2(word_freq)
        
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
        vocab, stoi, itos = vocabulaire2(word_freq)
        
        # Should be sorted alphabetically regardless of frequency
        expected_vocab = ["a", "m", "z"]
        assert vocab == expected_vocab
    
    def test_special_characters(self):
        """Test vocabulary with special characters in words"""
        word_freq = Counter({"word1": 1, "word_2": 2, "word-3": 1})
        vocab, stoi, itos = vocabulaire2(word_freq)
        
        expected_vocab = ["word-3", "word1", "word_2"]  # ASCII order
        assert vocab == expected_vocab
        
        # Verify mappings
        assert len(stoi) == len(itos) == len(vocab)
        for i, word in enumerate(vocab):
            assert stoi[word] == i
            assert itos[i] == word

