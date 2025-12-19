"""
Data loading and preprocessing utilities for Protein-to-Drug generation.
Loads CPI dataset and creates dataloaders for training.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, List, Dict
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CPIDataset(Dataset):
    """
    Custom PyTorch Dataset for Compound-Protein Interaction (CPI) data.
    Input: Protein embedding (from encoder output)
    Target: SMILES string (encoded as token indices)
    """
    
    def __init__(self, 
                 protein_embeddings: np.ndarray,
                 protein_ids: List[str],
                 smiles_list: List[str],
                 smiles_tokenizer,
                 protein_tokenizer=None,
                 max_smiles_len: int = 512,
                 max_protein_len: int = 1024,
                 device: str = 'cpu'):
        """
        Initialize CPI dataset.
        
        Args:
            protein_embeddings: Array of protein embeddings (n_proteins, embedding_dim)
            protein_ids: List of protein UniProt IDs
            smiles_list: List of SMILES strings paired with proteins
            smiles_tokenizer: SMILESTokenizer instance
            protein_tokenizer: ProteinTokenizer instance (optional)
            max_smiles_len: Maximum SMILES token sequence length
            max_protein_len: Maximum protein embedding length (for reshaping)
            device: Device to move tensors to
        """
        self.protein_embeddings = protein_embeddings
        self.protein_ids = protein_ids
        self.smiles_list = smiles_list
        self.smiles_tokenizer = smiles_tokenizer
        self.protein_tokenizer = protein_tokenizer
        self.max_smiles_len = max_smiles_len
        self.max_protein_len = max_protein_len
        self.device = device
        
        # Create protein_id to index mapping
        self.protein_id_to_idx = {pid: idx for idx, pid in enumerate(protein_ids)}
        
        assert len(smiles_list) == len(protein_embeddings), \
            f"SMILES list ({len(smiles_list)}) must match protein embeddings ({len(protein_embeddings)})"
        
        logger.info(f"Initialized CPIDataset with {len(self)} samples")
    
    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.smiles_list)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with keys: 'protein_embedding', 'smiles_input', 'smiles_target', 'smiles_mask'
        """
        # Get protein embedding
        protein_embedding = torch.FloatTensor(self.protein_embeddings[idx])
        
        # Get SMILES and encode
        smiles = self.smiles_list[idx]
        smiles_tokens = self.smiles_tokenizer.encode(smiles)
        
        # Prepare encoder input (with SOS token)
        sos_token = self.smiles_tokenizer.get_sos_token_idx()
        eos_token = self.smiles_tokenizer.get_eos_token_idx()
        pad_token = self.smiles_tokenizer.get_pad_token_idx()
        
        # Encoder input: SOS + SMILES tokens + EOS
        encoder_input = [sos_token] + smiles_tokens + [eos_token]
        
        # Decoder input: SOS + SMILES tokens
        decoder_input = [sos_token] + smiles_tokens
        
        # Target: SMILES tokens + EOS
        target = smiles_tokens + [eos_token]
        
        # Truncate if too long
        if len(encoder_input) > self.max_smiles_len:
            encoder_input = encoder_input[:self.max_smiles_len-1] + [eos_token]
        if len(decoder_input) > self.max_smiles_len:
            decoder_input = decoder_input[:self.max_smiles_len]
        if len(target) > self.max_smiles_len:
            target = target[:self.max_smiles_len]
        
        # Pad sequences
        encoder_input_padded = encoder_input + [pad_token] * (self.max_smiles_len - len(encoder_input))
        decoder_input_padded = decoder_input + [pad_token] * (self.max_smiles_len - len(decoder_input))
        target_padded = target + [pad_token] * (self.max_smiles_len - len(target))
        
        # Create attention masks
        encoder_mask = torch.BoolTensor([token != pad_token for token in encoder_input_padded])
        decoder_mask = torch.BoolTensor([token != pad_token for token in decoder_input_padded])
        
        return {
            'protein_embedding': protein_embedding,
            'encoder_input': torch.LongTensor(encoder_input_padded),
            'decoder_input': torch.LongTensor(decoder_input_padded),
            'label': torch.LongTensor(target_padded),
            'encoder_mask': encoder_mask,
            'decoder_mask': decoder_mask,
            'protein_id': self.protein_ids[idx]
        }


class CPIDataLoader:
    """Helper class to load CPI datasets from files."""
    
    @staticmethod
    def load_dataset(data_dir: str, 
                    protein_embeddings_file: str = "protein_embeddings.npz",
                    smiles_file: str = "smiles.smi",
                    protein_ids_file: str = "uniprot_ID.smi") -> Tuple[List[str], List[str]]:
        """
        Load SMILES and protein IDs from files.
        
        Args:
            data_dir: Directory containing data files
            protein_embeddings_file: Name of NPZ file with protein embeddings
            smiles_file: Name of file with SMILES strings (one per line)
            protein_ids_file: Name of file with protein IDs (one per line)
            
        Returns:
            Tuple of (smiles_list, protein_ids_list)
        """
        data_path = Path(data_dir)
        
        # Load SMILES
        smiles_path = data_path / smiles_file
        with open(smiles_path, 'r') as f:
            smiles_list = [line.strip() for line in f.readlines() if line.strip()]
        
        # Load protein IDs
        protein_ids_path = data_path / protein_ids_file
        with open(protein_ids_path, 'r') as f:
            protein_ids_list = [line.strip() for line in f.readlines() if line.strip()]
        
        logger.info(f"Loaded {len(smiles_list)} SMILES and {len(protein_ids_list)} protein IDs")
        
        return smiles_list, protein_ids_list
    
    @staticmethod
    def load_protein_embeddings(npz_file: str) -> Tuple[np.ndarray, List[str]]:
        """
        Load protein embeddings from NPZ file.
        
        Args:
            npz_file: Path to NPZ file with embeddings
            
        Returns:
            Tuple of (embeddings, protein_ids)
        """
        data = np.load(npz_file, allow_pickle=True)
        embeddings = data['embeddings']
        protein_ids = list(data['protein_ids'])
        
        logger.info(f"Loaded {len(protein_ids)} protein embeddings with shape {embeddings.shape}")
        
        return embeddings, protein_ids
    
    @staticmethod
    def create_dataloaders(smiles_list: List[str],
                          protein_ids_list: List[str],
                          protein_embeddings: np.ndarray,
                          protein_embedding_ids: List[str],
                          smiles_tokenizer,
                          batch_size: int = 32,
                          train_split: float = 0.8,
                          device: str = 'cpu',
                          shuffle_train: bool = True,
                          num_workers: int = 0) -> Tuple[DataLoader, DataLoader]:
        """
        Create train and validation dataloaders.
        
        Args:
            smiles_list: List of SMILES strings
            protein_ids_list: List of protein IDs
            protein_embeddings: Array of protein embeddings
            protein_embedding_ids: List of protein IDs corresponding to embeddings
            smiles_tokenizer: SMILESTokenizer instance
            batch_size: Batch size
            train_split: Fraction for training (rest goes to validation)
            device: Device to use
            shuffle_train: Whether to shuffle training data
            num_workers: Number of data loading workers
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Filter samples where protein embedding is available
        protein_embedding_id_set = set(protein_embedding_ids)
        valid_indices = [i for i, pid in enumerate(protein_ids_list) 
                        if pid in protein_embedding_id_set]
        
        filtered_smiles = [smiles_list[i] for i in valid_indices]
        filtered_protein_ids = [protein_ids_list[i] for i in valid_indices]
        
        # Map protein IDs to embedding indices
        protein_id_to_embedding_idx = {pid: idx for idx, pid in enumerate(protein_embedding_ids)}
        embedding_indices = [protein_id_to_embedding_idx[pid] for pid in filtered_protein_ids]
        filtered_embeddings = protein_embeddings[embedding_indices]
        
        logger.info(f"Filtered to {len(filtered_smiles)} samples with available embeddings")
        
        # Split into train/val
        n_samples = len(filtered_smiles)
        n_train = int(n_samples * train_split)
        
        train_indices = list(range(n_train))
        val_indices = list(range(n_train, n_samples))
        
        train_smiles = [filtered_smiles[i] for i in train_indices]
        train_protein_ids = [filtered_protein_ids[i] for i in train_indices]
        train_embeddings = filtered_embeddings[train_indices]
        
        val_smiles = [filtered_smiles[i] for i in val_indices]
        val_protein_ids = [filtered_protein_ids[i] for i in val_indices]
        val_embeddings = filtered_embeddings[val_indices]
        
        # Create datasets
        train_dataset = CPIDataset(
            protein_embeddings=train_embeddings,
            protein_ids=train_protein_ids,
            smiles_list=train_smiles,
            smiles_tokenizer=smiles_tokenizer,
            device=device
        )
        
        val_dataset = CPIDataset(
            protein_embeddings=val_embeddings,
            protein_ids=val_protein_ids,
            smiles_list=val_smiles,
            smiles_tokenizer=smiles_tokenizer,
            device=device
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            pin_memory=True if device == 'cuda' else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if device == 'cuda' else False
        )
        
        logger.info(f"Created dataloaders: {len(train_loader)} train batches, {len(val_loader)} val batches")
        
        return train_loader, val_loader


if __name__ == "__main__":
    # Example usage
    print("CPI Data loader initialized")
