import pytest
from src.text_utils.text_utils import tokenisation


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



