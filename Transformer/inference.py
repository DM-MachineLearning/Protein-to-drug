"""
Inference and drug generation module.
Generates drug SMILES from protein embeddings using the trained model.
"""

import torch
import numpy as np
from typing import List, Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DrugGenerator:
    """
    Generates drug SMILES strings from protein embeddings.
    Supports greedy decoding, beam search, and sampling.
    """
    
    def __init__(self, 
                 model,
                 smiles_tokenizer,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 max_length: int = 512):
        """
        Initialize drug generator.
        
        Args:
            model: Trained ProteinDrugTransformer model
            smiles_tokenizer: SMILESTokenizer for encoding/decoding
            device: Device to use
            max_length: Maximum length of generated SMILES
        """
        self.model = model.to(device)
        self.smiles_tokenizer = smiles_tokenizer
        self.device = device
        self.max_length = max_length
        self.model.eval()
    
    def greedy_decode(self, protein_embedding: torch.Tensor) -> str:
        """
        Generate drug SMILES using greedy decoding.
        
        Args:
            protein_embedding: Protein embedding tensor (1, embedding_dim)
            
        Returns:
            Generated SMILES string
        """
        protein_embedding = protein_embedding.to(self.device)
        
        # Initialize decoder input with SOS token
        sos_token = self.smiles_tokenizer.get_sos_token_idx()
        eos_token = self.smiles_tokenizer.get_eos_token_idx()
        pad_token = self.smiles_tokenizer.get_pad_token_idx()
        
        decoder_input = torch.LongTensor([[sos_token]]).to(self.device)
        generated_tokens = []
        
        with torch.no_grad():
            for _ in range(self.max_length):
                # Pad decoder input to max_length
                padded_input = torch.cat([
                    decoder_input,
                    torch.full((1, self.max_length - decoder_input.size(1)), pad_token, 
                              dtype=torch.long, device=self.device)
                ], dim=1)
                
                # Create mask
                mask = torch.ones((1, self.max_length), dtype=torch.bool, device=self.device)
                for i in range(decoder_input.size(1), self.max_length):
                    mask[0, i] = False
                
                # Forward pass
                logits = self.model(
                    protein_embedding=protein_embedding,
                    encoder_input=decoder_input,
                    decoder_input=padded_input,
                    encoder_mask=torch.ones((1, 1), dtype=torch.bool, device=self.device),
                    decoder_mask=mask
                )
                
                # Get next token (greedy)
                next_logits = logits[:, decoder_input.size(1) - 1, :]
                next_token = torch.argmax(next_logits, dim=-1)
                
                # Check for EOS token
                if next_token.item() == eos_token:
                    break
                
                generated_tokens.append(next_token.item())
                decoder_input = torch.cat([decoder_input, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
        
        # Decode tokens to SMILES
        smiles = self.smiles_tokenizer.decode(generated_tokens)
        return smiles
    
    def beam_search_decode(self, 
                          protein_embedding: torch.Tensor,
                          beam_width: int = 5,
                          length_penalty: float = 0.6) -> List[Tuple[str, float]]:
        """
        Generate drug SMILES using beam search.
        
        Args:
            protein_embedding: Protein embedding tensor (1, embedding_dim)
            beam_width: Width of beam search
            length_penalty: Length penalty for normalization
            
        Returns:
            List of (SMILES, score) tuples sorted by score
        """
        protein_embedding = protein_embedding.to(self.device)
        
        sos_token = self.smiles_tokenizer.get_sos_token_idx()
        eos_token = self.smiles_tokenizer.get_eos_token_idx()
        pad_token = self.smiles_tokenizer.get_pad_token_idx()
        
        # Initialize beam with SOS token
        # Each beam: (token_ids, score, is_finished)
        beams = [
            (torch.LongTensor([[sos_token]]).to(self.device), 0.0, False)
        ]
        
        completed = []
        
        with torch.no_grad():
            for step in range(self.max_length):
                candidates = []
                
                for token_ids, score, is_finished in beams:
                    if is_finished:
                        completed.append((token_ids, score))
                        continue
                    
                    # Pad input
                    padded_input = torch.cat([
                        token_ids,
                        torch.full((1, self.max_length - token_ids.size(1)), pad_token,
                                  dtype=torch.long, device=self.device)
                    ], dim=1)
                    
                    # Create mask
                    mask = torch.ones((1, self.max_length), dtype=torch.bool, device=self.device)
                    for i in range(token_ids.size(1), self.max_length):
                        mask[0, i] = False
                    
                    # Forward pass
                    logits = self.model(
                        protein_embedding=protein_embedding,
                        encoder_input=token_ids,
                        decoder_input=padded_input,
                        encoder_mask=torch.ones((1, 1), dtype=torch.bool, device=self.device),
                        decoder_mask=mask
                    )
                    
                    # Get next token logits
                    next_logits = logits[:, token_ids.size(1) - 1, :]
                    log_probs = torch.log_softmax(next_logits, dim=-1)
                    
                    # Get top beam_width tokens
                    top_log_probs, top_tokens = torch.topk(log_probs[0], beam_width)
                    
                    for log_prob, token in zip(top_log_probs, top_tokens):
                        new_ids = torch.cat([token_ids, token.unsqueeze(0).unsqueeze(0)], dim=1)
                        new_score = score + log_prob.item()
                        is_finished_new = token.item() == eos_token
                        candidates.append((new_ids, new_score, is_finished_new))
                
                # Keep only top beam_width candidates
                candidates.sort(key=lambda x: x[1], reverse=True)
                beams = candidates[:beam_width]
                
                # Check if all beams finished
                if all(is_finished for _, _, is_finished in beams):
                    break
        
        # Add remaining beams to completed
        for token_ids, score, _ in beams:
            completed.append((token_ids, score))
        
        # Sort by score and decode
        completed.sort(key=lambda x: x[1], reverse=True)
        results = []
        
        for token_ids, score in completed[:beam_width]:
            tokens = token_ids[0, 1:].tolist()  # Remove SOS token
            tokens = [t for t in tokens if t != eos_token and t != pad_token]
            smiles = self.smiles_tokenizer.decode(tokens)
            # Normalize score by length
            normalized_score = score / (len(tokens) ** length_penalty)
            results.append((smiles, normalized_score))
        
        return results
    
    def sample_decode(self, 
                     protein_embedding: torch.Tensor,
                     temperature: float = 1.0,
                     top_k: int = 0,
                     top_p: float = 0.9,
                     num_samples: int = 1) -> List[str]:
        """
        Generate drug SMILES using sampling with temperature and top-k/top-p filtering.
        
        Args:
            protein_embedding: Protein embedding tensor (1, embedding_dim)
            temperature: Sampling temperature (higher = more random)
            top_k: If > 0, only sample from top-k tokens
            top_p: If < 1.0, use nucleus sampling with cumulative probability top_p
            num_samples: Number of samples to generate
            
        Returns:
            List of generated SMILES strings
        """
        samples = []
        
        for _ in range(num_samples):
            protein_embedding_sample = protein_embedding.to(self.device)
            
            sos_token = self.smiles_tokenizer.get_sos_token_idx()
            eos_token = self.smiles_tokenizer.get_eos_token_idx()
            pad_token = self.smiles_tokenizer.get_pad_token_idx()
            
            decoder_input = torch.LongTensor([[sos_token]]).to(self.device)
            generated_tokens = []
            
            with torch.no_grad():
                for _ in range(self.max_length):
                    # Pad decoder input
                    padded_input = torch.cat([
                        decoder_input,
                        torch.full((1, self.max_length - decoder_input.size(1)), pad_token,
                                  dtype=torch.long, device=self.device)
                    ], dim=1)
                    
                    # Create mask
                    mask = torch.ones((1, self.max_length), dtype=torch.bool, device=self.device)
                    for i in range(decoder_input.size(1), self.max_length):
                        mask[0, i] = False
                    
                    # Forward pass
                    logits = self.model(
                        protein_embedding=protein_embedding_sample,
                        encoder_input=decoder_input,
                        decoder_input=padded_input,
                        encoder_mask=torch.ones((1, 1), dtype=torch.bool, device=self.device),
                        decoder_mask=mask
                    )
                    
                    # Get next token logits
                    next_logits = logits[:, decoder_input.size(1) - 1, :]
                    
                    # Apply temperature
                    next_logits = next_logits / temperature
                    
                    # Apply top-k filtering
                    if top_k > 0:
                        indices_to_remove = next_logits[0] < torch.topk(next_logits[0], top_k)[0][-1]
                        next_logits[0, indices_to_remove] = -float('Inf')
                    
                    # Apply top-p (nucleus) filtering
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_logits[0], descending=True)
                        cumsum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=0)
                        sorted_indices_to_remove = cumsum_probs > top_p
                        sorted_indices_to_remove[0] = False  # Keep at least one token
                        indices_to_remove = sorted_indices[sorted_indices_to_remove]
                        next_logits[0, indices_to_remove] = -float('Inf')
                    
                    # Sample
                    probs = torch.softmax(next_logits, dim=-1)
                    next_token = torch.multinomial(probs[0], num_samples=1)
                    
                    # Check for EOS token
                    if next_token.item() == eos_token:
                        break
                    
                    generated_tokens.append(next_token.item())
                    decoder_input = torch.cat([decoder_input, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
            
            # Decode tokens to SMILES
            smiles = self.smiles_tokenizer.decode(generated_tokens)
            samples.append(smiles)
        
        return samples
    
    def generate_batch(self, 
                      protein_embeddings: np.ndarray,
                      method: str = 'greedy',
                      **kwargs) -> List[str]:
        """
        Generate SMILES for a batch of protein embeddings.
        
        Args:
            protein_embeddings: Array of protein embeddings (batch_size, embedding_dim)
            method: 'greedy', 'beam_search', or 'sample'
            **kwargs: Additional arguments for generation method
            
        Returns:
            List of generated SMILES strings
        """
        results = []
        
        for embedding in protein_embeddings:
            embedding_tensor = torch.FloatTensor(embedding).unsqueeze(0)
            
            if method == 'greedy':
                smiles = self.greedy_decode(embedding_tensor)
                results.append(smiles)
            elif method == 'beam_search':
                beam_results = self.beam_search_decode(embedding_tensor, **kwargs)
                results.append(beam_results[0][0] if beam_results else "")
            elif method == 'sample':
                num_samples = kwargs.get('num_samples', 1)
                samples = self.sample_decode(embedding_tensor, num_samples=num_samples, **kwargs)
                results.extend(samples)
            else:
                raise ValueError(f"Unknown method: {method}")
        
        return results


class MolecularValidator:
    """Validates generated SMILES for chemical validity."""
    
    @staticmethod
    def validate_smiles(smiles: str) -> Tuple[bool, str]:
        """
        Validate SMILES string.
        
        Args:
            smiles: SMILES string to validate
            
        Returns:
            Tuple of (is_valid, message)
        """
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                return True, "Valid SMILES"
            else:
                return False, "RDKit could not parse SMILES"
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    @staticmethod
    def canonicalize_smiles(smiles: str) -> str:
        """
        Canonicalize SMILES string using RDKit.
        
        Args:
            smiles: SMILES string
            
        Returns:
            Canonicalized SMILES
        """
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                return Chem.MolToSmiles(mol)
            else:
                return smiles
        except Exception as e:
            logger.warning(f"Could not canonicalize SMILES: {e}")
            return smiles
    
    @staticmethod
    def calculate_properties(smiles: str) -> Dict:
        """
        Calculate molecular properties.
        
        Args:
            smiles: SMILES string
            
        Returns:
            Dictionary of properties
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {}
            
            return {
                'molecular_weight': Descriptors.MolWt(mol),
                'logp': Descriptors.MolLogP(mol),
                'num_h_donors': Descriptors.NumHDonors(mol),
                'num_h_acceptors': Descriptors.NumHAcceptors(mol),
                'num_rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                'tpsa': Descriptors.TPSA(mol),
            }
        except Exception as e:
            logger.warning(f"Could not calculate properties: {e}")
            return {}


if __name__ == "__main__":
    print("Drug generator module initialized")
