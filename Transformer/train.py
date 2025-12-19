"""
Training pipeline for Protein-to-Drug generation using Transformer.
Handles training loops, validation, and checkpoint management.
"""

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
import math
from tqdm import tqdm
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProteinDrugTransformer(nn.Module):
    """
    Protein-to-Drug generation model using Transformer.
    Encoder takes protein embeddings, decoder generates drug SMILES.
    """
    
    def __init__(self, 
                 transformer,
                 protein_embedding_dim: int,
                 protein_projection_dim: int = 512):
        """
        Initialize model.
        
        Args:
            transformer: Transformer model from model.py
            protein_embedding_dim: Dimension of protein embeddings
            protein_projection_dim: Dimension to project proteins to (should match transformer d_model)
        """
        super().__init__()
        self.transformer = transformer
        self.protein_projection = nn.Linear(protein_embedding_dim, protein_projection_dim)
    
    def encode_protein(self, protein_embedding: torch.Tensor) -> torch.Tensor:
        """
        Encode protein embedding.
        
        Args:
            protein_embedding: (batch_size, protein_embedding_dim)
            
        Returns:
            Encoded protein: (batch_size, seq_len=1, d_model)
        """
        projected = self.protein_projection(protein_embedding)  # (batch, d_model)
        # Determine an appropriate encoder sequence length to expand into.
        # If the transformer was built with a very small src_seq_len (e.g. 1),
        # prefer expanding to the decoder's positional length so cross-attention
        # shapes align (commonly tgt_seq_len like 512). This avoids shape
        # mismatches in attention when the encoder provides a single vector.
        try:
            src_seq_len = getattr(self.transformer.src_pos, 'seq_len', None)
            tgt_seq_len = getattr(self.transformer.tgt_pos, 'seq_len', None)
        except Exception:
            src_seq_len = None
            tgt_seq_len = None

        if src_seq_len is None or src_seq_len <= 1:
            seq_len_to_use = tgt_seq_len if (tgt_seq_len is not None and tgt_seq_len > 1) else 1
        else:
            seq_len_to_use = src_seq_len

        # Repeat projected vector across the chosen sequence length so decoder cross-attention has proper shape
        projected_expanded = projected.unsqueeze(1).expand(-1, seq_len_to_use, -1).contiguous()
        return projected_expanded  # (batch, seq_len_to_use, d_model)
    
    def forward(self, 
                protein_embedding: torch.Tensor,
                encoder_input: torch.Tensor,
                decoder_input: torch.Tensor,
                encoder_mask: torch.Tensor,
                decoder_mask: torch.Tensor):
        """
        Forward pass.
        
        Args:
            protein_embedding: (batch_size, protein_embedding_dim)
            encoder_input: (batch_size, smiles_seq_len) - not used, using protein as encoder
            decoder_input: (batch_size, smiles_seq_len)
            encoder_mask: (batch_size, 1) for protein
            decoder_mask: (batch_size, smiles_seq_len)
            
        Returns:
            Logits: (batch_size, smiles_seq_len, vocab_size)
        """
        # Encode protein
        encoder_output = self.encode_protein(protein_embedding)  # (batch, 1, d_model)
        
        # Prepare masks
        # Encoder mask: (batch, 1, 1, 1)
        if encoder_mask is not None:
            encoder_mask = encoder_mask.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        
        # Decoder mask: (batch, 1, seq_len, seq_len) - causal mask
        if decoder_mask is not None:
            batch_size, seq_len = decoder_input.shape
            decoder_mask_2d = torch.triu(torch.ones(seq_len, seq_len, device=decoder_input.device), diagonal=1).bool()
            decoder_mask_2d = ~decoder_mask_2d  # Invert for attention
            decoder_mask_2d = decoder_mask_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        else:
            decoder_mask_2d = None
        
        # Decode
        # transformer.decode signature: (encoder_output, src_mask, tgt, tgt_mask)
        decoder_output = self.transformer.decode(
            encoder_output,
            encoder_mask,
            decoder_input,
            decoder_mask_2d
        )  # (batch, seq_len, d_model)
        
        # Project to vocabulary
        logits = self.transformer.project(decoder_output)  # (batch, seq_len, vocab_size)
        
        return logits


class TrainingConfig:
    """Training configuration."""
    
    def __init__(self, 
                 batch_size: int = 32,
                 epochs: int = 100,
                 learning_rate: float = 3e-4,
                 weight_decay: float = 1e-5,
                 warmup_steps: int = 4000,
                 gradient_clip_val: float = 1.0,
                 label_smoothing: float = 0.1,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """Initialize config."""
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.gradient_clip_val = gradient_clip_val
        self.label_smoothing = label_smoothing
        self.device = device
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'warmup_steps': self.warmup_steps,
            'gradient_clip_val': self.gradient_clip_val,
            'label_smoothing': self.label_smoothing,
        }


class Trainer:
    """Trainer class for model training and validation."""
    
    def __init__(self,
                 model: nn.Module,
                 config: TrainingConfig,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 output_dir: str = "./checkpoints",
                 checkpoint_every: int = 10):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            config: Training configuration
            train_loader: Training dataloader
            val_loader: Validation dataloader
            output_dir: Directory to save checkpoints
        """
        self.model = model.to(config.device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.checkpoint_every = checkpoint_every
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.98)
        )
        
        # Learning rate scheduler with warmup
        self.scheduler = self._create_scheduler()
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=config.label_smoothing,
            ignore_index=0  # Ignore padding tokens
        )
        
        # Tracking
        self.best_val_loss = float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        logger.info("Trainer initialized")
    
    def _create_scheduler(self):
        """Create learning rate scheduler with warmup."""
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return float(step) / float(max(1, self.config.warmup_steps))
            return max(0.0, float(self.config.epochs - step) / float(max(1, self.config.epochs - self.config.warmup_steps)))
        
        return LambdaLR(self.optimizer, lr_lambda)
    
    def train_epoch(self) -> float:
        """
        Train for one epoch.
        
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        pbar = tqdm(self.train_loader, desc="Training", leave=True)
        
        for batch in pbar:
            # Move batch to device
            batch = {k: v.to(self.config.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            logits = self.model(
                protein_embedding=batch['protein_embedding'],
                encoder_input=batch['encoder_input'],
                decoder_input=batch['decoder_input'],
                encoder_mask=batch['encoder_mask'],
                decoder_mask=batch['decoder_mask']
            )
            
            # Reshape for loss calculation
            # logits: (batch, seq_len, vocab_size)
            # label: (batch, seq_len)
            logits_flat = logits.view(-1, logits.size(-1))
            labels_flat = batch['label'].view(-1)
            
            # Calculate loss
            loss = self.criterion(logits_flat, labels_flat)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config.gradient_clip_val > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_val)
            
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            n_batches += 1
            
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / n_batches
        return avg_loss
    
    def validate(self) -> float:
        """
        Validate model.
        
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validating", leave=True)
            
            for batch in pbar:
                # Move batch to device
                batch = {k: v.to(self.config.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                logits = self.model(
                    protein_embedding=batch['protein_embedding'],
                    encoder_input=batch['encoder_input'],
                    decoder_input=batch['decoder_input'],
                    encoder_mask=batch['encoder_mask'],
                    decoder_mask=batch['decoder_mask']
                )
                
                # Reshape for loss calculation
                logits_flat = logits.view(-1, logits.size(-1))
                labels_flat = batch['label'].view(-1)
                
                # Calculate loss
                loss = self.criterion(logits_flat, labels_flat)
                
                total_loss += loss.item()
                n_batches += 1
                
                pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / n_batches
        return avg_loss
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config.to_dict(),
            'best_val_loss': self.best_val_loss,
        }
        
        save_path = self.output_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, save_path)
        logger.info(f"Saved checkpoint to {save_path}")
        
        if is_best:
            best_path = self.output_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint['epoch']
    
    def train(self, start_epoch: int = 0):
        """
        Full training loop.
        
        Args:
            start_epoch: Epoch to start from (for resuming)
        """
        logger.info(f"Starting training for {self.config.epochs} epochs")
        
        for epoch in range(start_epoch, self.config.epochs):
            logger.info(f"\nEpoch {epoch+1}/{self.config.epochs}")
            
            # Train
            train_loss = self.train_epoch()
            logger.info(f"Train Loss: {train_loss:.4f}")
            
            # Validate
            val_loss = self.validate()
            logger.info(f"Val Loss: {val_loss:.4f}")
            
            # Track history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            # Save checkpoint periodically and best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, is_best=True)

            # Periodic checkpointing to allow resume without storing every epoch
            if (epoch + 1) % getattr(self, 'checkpoint_every', 10) == 0:
                self.save_checkpoint(epoch, is_best=False)
            
            # Save history
            history_path = self.output_dir / "training_history.json"
            with open(history_path, 'w') as f:
                json.dump(self.training_history, f, indent=2)
        
        logger.info("Training completed!")
    
    def get_history(self) -> Dict:
        """Get training history."""
        return self.training_history


if __name__ == "__main__":
    print("Training pipeline initialized")

