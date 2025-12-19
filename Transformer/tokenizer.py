"""
SMILES Tokenizer and Vocabulary Builder
Tokenizes SMILES strings into discrete tokens for model input.
"""

import re
from typing import List, Dict, Set
from collections import Counter
import pickle
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SMILESTokenizer:
    """
    Tokenizes SMILES strings into tokens.
    Uses regex-based approach to identify chemical tokens.
    """
    
    # SMILES token regex pattern
    # Matches: atoms, bonds, branches, charges, etc.
    SMILES_PATTERN = r"""
        (\[(?:[^\]]*)\]|  # Bracket atoms
        Cl|Br|           # Two-character atoms
        [#%()+=\\-\\[\\]/,@]|  # Bonds and notation
        [A-Z][a-z]?|     # Element symbols
        [0-9]+|           # Numbers
        [cnos])          # Lowercase heteroatoms
    """
    
    SPECIAL_TOKENS = {
        '<PAD>': 0,
        '<SOS>': 1,
        '<EOS>': 2,
        '<UNK>': 3,
    }
    
    def __init__(self, vocab: Dict[str, int] = None):
        """
        Initialize tokenizer.
        
        Args:
            vocab: Token to index mapping. If None, uses only special tokens.
        """
        self.vocab = vocab or self.SPECIAL_TOKENS.copy()
        self.idx_to_token = {v: k for k, v in self.vocab.items()}
    
    @staticmethod
    def tokenize(smiles: str) -> List[str]:
        """
        Tokenize a SMILES string.
        
        Args:
            smiles: SMILES string
            
        Returns:
            List of tokens
        """
        pattern = re.compile(SMILESTokenizer.SMILES_PATTERN, re.VERBOSE)
        tokens = pattern.findall(smiles)
        return tokens
    
    def build_vocab(self, smiles_list: List[str], min_freq: int = 1) -> Dict[str, int]:
        """
        Build vocabulary from SMILES strings.
        
        Args:
            smiles_list: List of SMILES strings
            min_freq: Minimum frequency threshold for tokens
            
        Returns:
            Token to index mapping
        """
        token_counter = Counter()
        
        logger.info(f"Tokenizing {len(smiles_list)} SMILES strings...")
        for smiles in smiles_list:
            tokens = self.tokenize(smiles)
            token_counter.update(tokens)
        
        # Filter by frequency
        vocab = self.SPECIAL_TOKENS.copy()
        next_idx = len(vocab)
        
        for token, freq in token_counter.most_common():
            if freq >= min_freq:
                vocab[token] = next_idx
                next_idx += 1
        
        self.vocab = vocab
        self.idx_to_token = {v: k for k, v in vocab.items()}
        
        logger.info(f"Built vocabulary with {len(vocab)} tokens")
        return vocab
    
    def encode(self, smiles: str) -> List[int]:
        """
        Encode SMILES string to token indices.
        
        Args:
            smiles: SMILES string
            
        Returns:
            List of token indices
        """
        tokens = self.tokenize(smiles)
        indices = [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]
        return indices
    
    def decode(self, indices: List[int]) -> str:
        """
        Decode token indices back to SMILES string.
        
        Args:
            indices: List of token indices
            
        Returns:
            SMILES string
        """
        tokens = [self.idx_to_token.get(idx, '<UNK>') for idx in indices]
        # Remove special tokens and join
        tokens = [t for t in tokens if t not in ['<PAD>', '<SOS>', '<EOS>', '<UNK>']]
        return ''.join(tokens)
    
    def save(self, filepath: str):
        """Save tokenizer to file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.vocab, f)
        logger.info(f"Saved tokenizer to {filepath}")
    
    @staticmethod
    def load(filepath: str) -> 'SMILESTokenizer':
        """Load tokenizer from file."""
        with open(filepath, 'rb') as f:
            vocab = pickle.load(f)
        logger.info(f"Loaded tokenizer from {filepath}")
        return SMILESTokenizer(vocab)
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.vocab)
    
    def get_pad_token_idx(self) -> int:
        """Get padding token index."""
        return self.vocab['<PAD>']
    
    def get_sos_token_idx(self) -> int:
        """Get start-of-sequence token index."""
        return self.vocab['<SOS>']
    
    def get_eos_token_idx(self) -> int:
        """Get end-of-sequence token index."""
        return self.vocab['<EOS>']


class ProteinTokenizer:
    """
    Simple tokenizer for protein sequences.
    Maps 20 standard amino acids to indices.
    """
    
    AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
    
    SPECIAL_TOKENS = {
        '<PAD>': 0,
        '<SOS>': 1,
        '<EOS>': 2,
        '<UNK>': 3,
    }
    
    def __init__(self):
        """Initialize protein tokenizer."""
        self.vocab = self.SPECIAL_TOKENS.copy()
        next_idx = len(self.vocab)
        for aa in self.AMINO_ACIDS:
            self.vocab[aa] = next_idx
            next_idx += 1
        self.idx_to_token = {v: k for k, v in self.vocab.items()}
    
    def encode(self, sequence: str) -> List[int]:
        """
        Encode protein sequence to token indices.
        
        Args:
            sequence: Protein sequence string
            
        Returns:
            List of token indices
        """
        indices = [self.vocab.get(aa, self.vocab['<UNK>']) for aa in sequence.upper()]
        return indices
    
    def decode(self, indices: List[int]) -> str:
        """
        Decode token indices back to protein sequence.
        
        Args:
            indices: List of token indices
            
        Returns:
            Protein sequence string
        """
        tokens = [self.idx_to_token.get(idx, '<UNK>') for idx in indices]
        # Remove special tokens
        tokens = [t for t in tokens if t not in ['<PAD>', '<SOS>', '<EOS>', '<UNK>']]
        return ''.join(tokens)
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.vocab)
    
    def get_pad_token_idx(self) -> int:
        return self.vocab['<PAD>']
    
    def get_sos_token_idx(self) -> int:
        return self.vocab['<SOS>']
    
    def get_eos_token_idx(self) -> int:
        return self.vocab['<EOS>']


if __name__ == "__main__":
    # Test SMILES tokenizer
    tokenizer = SMILESTokenizer()
    test_smiles = ["CCO", "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"]
    
    print("Testing SMILES tokenizer:")
    for smiles in test_smiles:
        tokens = tokenizer.tokenize(smiles)
        print(f"SMILES: {smiles}")
        print(f"Tokens: {tokens}")
    
    # Build vocabulary
    vocab = tokenizer.build_vocab(test_smiles)
    print(f"\nVocabulary size: {len(vocab)}")
    
    # Test encoding/decoding
    encoded = tokenizer.encode(test_smiles[0])
    decoded = tokenizer.decode(encoded)
    print(f"\nOriginal: {test_smiles[0]}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    
    # Test protein tokenizer
    print("\n\nTesting protein tokenizer:")
    protein_tokenizer = ProteinTokenizer()
    test_sequence = "MKFLKFSLLTAVLLSVVFAFSSCGDDDDTG"
    encoded = protein_tokenizer.encode(test_sequence)
    decoded = protein_tokenizer.decode(encoded)
    print(f"Original: {test_sequence}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    print(f"Vocab size: {protein_tokenizer.get_vocab_size()}")
