

from typing import List
import re


class SimpleTokenizer:
    """Simple tokenizer for language modeling"""
    
    def __init__(self, vocab_size: int = 15000):
        self.vocab_size = vocab_size
        self.word_to_id = {'<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3}
        self.id_to_word = {0: '<pad>', 1: '<unk>', 2: '<bos>', 3: '<eos>'}
        self.next_id = 4
    
    def build_vocab(self, texts: List[str]):
        """Build vocabulary from texts"""
        word_counts = {}
        for text in texts:
            words = self._tokenize(text)
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Add most frequent words
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        for word, count in sorted_words[:self.vocab_size - 4]:
            if word not in self.word_to_id:
                self.word_to_id[word] = self.next_id
                self.id_to_word[self.next_id] = word
                self.next_id += 1
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        # Basic preprocessing
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s.,!?;:]', ' ', text)
        words = text.split()
        return words
    
    def encode(self, text: str, max_length: int = 512) -> List[int]:
        """Encode text to token IDs"""
        words = self._tokenize(text)
        tokens = [self.word_to_id.get('<bos>', 2)]  # Start with BOS token
        
        for word in words[:max_length-2]:  # Leave space for BOS and EOS
            token_id = self.word_to_id.get(word, self.word_to_id.get('<unk>', 1))
            tokens.append(token_id)
        
        tokens.append(self.word_to_id.get('<eos>', 3))  # End with EOS token
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text"""
        words = []
        for token_id in token_ids:
            word = self.id_to_word.get(token_id, '<unk>')
            if word not in ['<pad>', '<bos>', '<eos>']:  # Skip special tokens
                words.append(word)
        return ' '.join(words)
