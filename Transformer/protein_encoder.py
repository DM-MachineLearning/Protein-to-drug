"""
Protein Characterization Module
Encodes protein sequences (from UniProt IDs) into fixed-size embeddings using protein language models.
Requires ProtTrans pretrained models - download from https://github.com/agemagician/ProtTrans
"""

import numpy as np
import torch
import requests
from typing import Dict, List, Tuple
import os
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProteinEncoder:
    """
    Encodes protein sequences to embeddings using various methods.
    Supports ESM2, ProtBERT, or other protein language models.
    """
    
    def __init__(self, model_name: str = "esm2", device: str = None):
        """
        Initialize protein encoder.
        
        Args:
            model_name: Type of model ('esm2', 'protbert', or 'prottrans')
            device: torch device ('cuda' or 'cpu')
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.alphabet = None
        self._load_model()
    
    def _load_model(self):
        """Load pretrained protein language model."""
        try:
            if self.model_name == "esm2":
                self._load_esm2()
            elif self.model_name == "protbert":
                self._load_protbert()
            elif self.model_name == "prottrans":
                self._load_prottrans()
            else:
                raise ValueError(f"Unknown model: {self.model_name}")
        except Exception as e:
            logger.warning(f"Could not load {self.model_name}: {e}")
            logger.info("Falling back to simple one-hot encoding")
            self.model = None
    
    def _load_esm2(self):
        """Load ESM2 model from Meta."""
        try:
            import esm
            self.model, self.alphabet = esm.pretrained.esmfold_structure_module_esmif1()
            # Use ESM2 3B model (balanced size/performance)
            self.model, self.alphabet = esm.pretrained.esmif1_structure_prediction_3b_esmif1()
            self.model.eval()
            self.model.to(self.device)
            logger.info("Loaded ESM2 model successfully")
        except ImportError:
            logger.warning("ESM not installed. Install with: pip install fair-esm")
            self.model = None
    
    def _load_protbert(self):
        """Load ProtBERT model from transformers."""
        try:
            from transformers import AutoTokenizer, AutoModel
            model_id = "Rostlab/prot_bert"
            self.alphabet = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModel.from_pretrained(model_id)
            self.model.eval()
            self.model.to(self.device)
            logger.info("Loaded ProtBERT model successfully")
        except ImportError:
            logger.warning("Transformers not installed. Install with: pip install transformers")
            self.model = None
    
    def _load_prottrans(self):
        """Load ProtTrans model."""
        try:
            from transformers import T5Tokenizer, T5EncoderModel
            model_id = "Rostlab/prot_t5_xl_half_uniref50-enc"
            self.alphabet = T5Tokenizer.from_pretrained(model_id, do_lower_case=False)
            self.model = T5EncoderModel.from_pretrained(model_id)
            self.model.eval()
            self.model.to(self.device)
            logger.info("Loaded ProtTrans model successfully")
        except ImportError:
            logger.warning("Transformers not installed. Install with: pip install transformers")
            self.model = None
    
    def fetch_sequence_from_uniprot(self, uniprot_id: str) -> str:
        """
        Fetch protein sequence from UniProt API.
        
        Args:
            uniprot_id: UniProt accession ID
            
        Returns:
            Protein sequence in FASTA format
        """
        try:
            url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                lines = response.text.strip().split('\n')
                # Skip header line
                sequence = ''.join(lines[1:])
                return sequence
            else:
                logger.warning(f"Could not fetch sequence for {uniprot_id} (status: {response.status_code})")
                return None
        except Exception as e:
            logger.warning(f"Error fetching {uniprot_id}: {e}")
            return None
    
    def load_sequence_cache(self, cache_file: str = "protein_sequences.npz") -> Dict[str, str]:
        """
        Load cached protein sequences.
        
        Args:
            cache_file: Path to NPZ file with cached sequences
            
        Returns:
            Dictionary of {uniprot_id: sequence}
        """
        if os.path.exists(cache_file):
            cache = np.load(cache_file, allow_pickle=True)
            return dict(cache['sequences'].item())
        return {}
    
    def save_sequence_cache(self, sequences: Dict[str, str], cache_file: str = "protein_sequences.npz"):
        """Save protein sequences to cache."""
        np.savez_compressed(cache_file, sequences=np.array([sequences], dtype=object))
        logger.info(f"Saved {len(sequences)} sequences to {cache_file}")
    
    def encode_sequence(self, sequence: str) -> np.ndarray:
        """
        Encode a single protein sequence to embedding.
        
        Args:
            sequence: Protein sequence string
            
        Returns:
            Embedding vector
        """
        if not sequence:
            return None
        
        if self.model is None:
            # Fallback: simple one-hot encoding
            return self._onehot_encode(sequence)
        
        try:
            if self.model_name == "esm2":
                return self._encode_esm2(sequence)
            elif self.model_name == "protbert":
                return self._encode_protbert(sequence)
            elif self.model_name == "prottrans":
                return self._encode_prottrans(sequence)
        except Exception as e:
            logger.warning(f"Error encoding sequence: {e}. Using one-hot fallback.")
            return self._onehot_encode(sequence)
    
    def _encode_esm2(self, sequence: str) -> np.ndarray:
        """Encode using ESM2."""
        with torch.no_grad():
            tokens = self.alphabet.encode(sequence)
            tokens = torch.tensor(tokens, device=self.device).unsqueeze(0)
            results = self.model(tokens, repr_layers=[33])
            embedding = results["representations"][33].mean(dim=1)
        return embedding.cpu().numpy().flatten()
    
    def _encode_protbert(self, sequence: str) -> np.ndarray:
        """Encode using ProtBERT."""
        # Add spaces between amino acids for ProtBERT
        sequence_processed = " ".join(sequence)
        
        with torch.no_grad():
            inputs = self.alphabet(sequence_processed, return_tensors="pt", padding=True, truncation=True, max_length=2048)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1)
        
        return embedding.cpu().numpy().flatten()
    
    def _encode_prottrans(self, sequence: str) -> np.ndarray:
        """Encode using ProtTrans."""
        # Add spaces between amino acids for ProtTrans
        sequence_processed = " ".join(sequence)
        
        with torch.no_grad():
            ids = self.alphabet.encode(sequence_processed, add_special_tokens=True)
            input_ids = torch.tensor(ids, device=self.device).unsqueeze(0)
            attention_mask = torch.ones_like(input_ids)
            embedding = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            embedding = embedding.mean(dim=1)
        
        return embedding.cpu().numpy().flatten()
    
    def _onehot_encode(self, sequence: str, max_len: int = 1024) -> np.ndarray:
        """
        Simple one-hot encoding fallback.
        Amino acids: ACDEFGHIKLMNPQRSTVWY (20 standard)
        """
        aa_list = "ACDEFGHIKLMNPQRSTVWY"
        aa_to_idx = {aa: i for i, aa in enumerate(aa_list)}
        
        # Truncate or pad sequence
        if len(sequence) > max_len:
            sequence = sequence[:max_len]
        
        # One-hot encode
        encoding = np.zeros((max_len, len(aa_list)), dtype=np.float32)
        for i, aa in enumerate(sequence):
            if aa in aa_to_idx:
                encoding[i, aa_to_idx[aa]] = 1.0
        
        return encoding.flatten()
    
    def encode_batch(self, sequences: List[str], batch_size: int = 4) -> np.ndarray:
        """
        Encode a batch of sequences.
        
        Args:
            sequences: List of protein sequences
            batch_size: Batch size for processing
            
        Returns:
            Array of embeddings (n_sequences, embedding_dim)
        """
        embeddings = []
        
        for i in tqdm(range(0, len(sequences), batch_size), desc="Encoding proteins"):
            batch = sequences[i:i+batch_size]
            batch_embeddings = [self.encode_sequence(seq) for seq in batch]
            batch_embeddings = [e for e in batch_embeddings if e is not None]
            embeddings.extend(batch_embeddings)
        
        return np.array(embeddings, dtype=np.float32)


class ProteinDatasetBuilder:
    """
    Builds NPZ dataset files with protein embeddings for training.
    """
    
    def __init__(self, encoder: ProteinEncoder):
        """
        Initialize dataset builder.
        
        Args:
            encoder: ProteinEncoder instance
        """
        self.encoder = encoder
    
    def build_from_uniprot_ids(self, uniprot_ids: List[str], 
                              output_file: str = "protein_embeddings.npz",
                              use_cache: bool = True) -> Tuple[np.ndarray, List[str]]:
        """
        Build dataset from UniProt IDs.
        
        Args:
            uniprot_ids: List of UniProt accession IDs
            output_file: Path to save NPZ file
            use_cache: Whether to use cached sequences
            
        Returns:
            Tuple of (embeddings, valid_ids)
        """
        # Load cached sequences if available
        cache_file = output_file.replace('.npz', '_sequences.npz')
        sequences_cache = self.encoder.load_sequence_cache(cache_file) if use_cache else {}
        
        sequences = {}
        missing_ids = []
        
        logger.info(f"Fetching sequences for {len(uniprot_ids)} UniProt IDs...")
        
        for uniprot_id in tqdm(uniprot_ids, desc="Fetching sequences"):
            if uniprot_id in sequences_cache:
                sequences[uniprot_id] = sequences_cache[uniprot_id]
            else:
                seq = self.encoder.fetch_sequence_from_uniprot(uniprot_id)
                if seq:
                    sequences[uniprot_id] = seq
                else:
                    missing_ids.append(uniprot_id)
        
        # Update cache
        sequences_cache.update(sequences)
        self.encoder.save_sequence_cache(sequences_cache, cache_file)
        
        logger.info(f"Successfully fetched {len(sequences)} sequences, {len(missing_ids)} missing")
        
        # Encode sequences
        valid_ids = list(sequences.keys())
        sequence_list = [sequences[uid] for uid in valid_ids]
        
        logger.info("Encoding sequences to embeddings...")
        embeddings = self.encoder.encode_batch(sequence_list)
        
        # Save to NPZ
        np.savez_compressed(output_file, 
                          embeddings=embeddings, 
                          protein_ids=np.array(valid_ids),
                          metadata={'model': self.encoder.model_name, 'num_proteins': len(valid_ids)})
        
        logger.info(f"Saved embeddings to {output_file}")
        logger.info(f"Embedding shape: {embeddings.shape}")
        
        return embeddings, valid_ids
    
    def load_embeddings(self, npz_file: str) -> Tuple[np.ndarray, List[str]]:
        """Load embeddings from NPZ file."""
        data = np.load(npz_file, allow_pickle=True)
        embeddings = data['embeddings']
        protein_ids = list(data['protein_ids'])
        return embeddings, protein_ids


if __name__ == "__main__":
    # Example usage
    encoder = ProteinEncoder(model_name="protbert")
    
    # Test single sequence encoding
    test_sequence = "MKFLKFSLLTAVLLSVVFAFSSCGDDDDTG"
    embedding = encoder.encode_sequence(test_sequence)
    print(f"Embedding shape: {embedding.shape}")
    
    # Build dataset from UniProt IDs
    test_ids = ["P12345", "P67890"]  # Example UniProt IDs
    builder = ProteinDatasetBuilder(encoder)
    embeddings, valid_ids = builder.build_from_uniprot_ids(test_ids)
