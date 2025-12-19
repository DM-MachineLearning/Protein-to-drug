"""
Main training script for Protein-to-Drug generation.
Orchestrates data loading, model training, and inference.

Usage:
    python main.py --stage preprocess  # Prepare protein embeddings
    python main.py --stage train        # Train the model
    python main.py --stage generate     # Generate drugs from proteins
"""

import argparse
import logging
import numpy as np
import torch
from pathlib import Path
import json

# Import custom modules
from model import build_transformer
from protein_encoder import ProteinEncoder, ProteinDatasetBuilder
from tokenizer import SMILESTokenizer, ProteinTokenizer
from data_loader import CPIDataLoader, CPIDataset
from train import ProteinDrugTransformer, TrainingConfig, Trainer
from inference import DrugGenerator, MolecularValidator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    'data_dir': '../CPI/CPI',
    'output_dir': './results',
    'checkpoint_dir': './checkpoints',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    # Model hyperparameters
    'd_model': 512,
    'num_layers': 6,
    'num_heads': 8,
    'd_ff': 2048,
    'dropout': 0.1,
    'max_smiles_len': 512,
    'max_protein_embedding_len': 1024,
    
    # Training hyperparameters
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 3e-4,
    'weight_decay': 1e-5,
    'warmup_steps': 4000,
    'gradient_clip_val': 1.0,
    'label_smoothing': 0.1,
    'train_split': 0.8,
    
    # Protein encoding
    'protein_model': 'protbert',  # or 'esm2', 'prottrans'
    'protein_embeddings_file': 'protein_embeddings.npz',
    
    # Generation
    'generation_method': 'beam_search',  # or 'greedy', 'sample'
    'beam_width': 5,
}


def preprocess_proteins():
    """
    Stage 1: Preprocess proteins
    Fetch protein sequences from UniProt and create embeddings.
    """
    logger.info("=" * 80)
    logger.info("STAGE 1: PREPROCESSING PROTEINS")
    logger.info("=" * 80)
    
    # Load SMILES and protein IDs from data
    logger.info(f"Loading data from {CONFIG['data_dir']}")
    smiles_list, protein_ids_list = CPIDataLoader.load_dataset(CONFIG['data_dir'])
    
    # Get unique protein IDs
    unique_protein_ids = list(set(protein_ids_list))
    logger.info(f"Found {len(unique_protein_ids)} unique proteins")
    logger.info(f"Found {len(smiles_list)} SMILES compounds")
    
    # Initialize protein encoder
    logger.info(f"Initializing protein encoder: {CONFIG['protein_model']}")
    encoder = ProteinEncoder(model_name=CONFIG['protein_model'], device=CONFIG['device'])
    
    # Build dataset
    logger.info("Building protein embeddings dataset...")
    builder = ProteinDatasetBuilder(encoder)
    
    output_path = Path(CONFIG['output_dir'])
    output_path.mkdir(exist_ok=True)
    embeddings_file = output_path / CONFIG['protein_embeddings_file']
    
    embeddings, valid_ids = builder.build_from_uniprot_ids(
        unique_protein_ids,
        output_file=str(embeddings_file),
        use_cache=True
    )
    
    logger.info(f"Successfully created {len(valid_ids)} protein embeddings")
    logger.info(f"Embeddings saved to {embeddings_file}")
    
    return str(embeddings_file), smiles_list, protein_ids_list


def prepare_data(embeddings_file):
    """Prepare SMILES tokenizer and build vocabulary."""
    logger.info("=" * 80)
    logger.info("PREPARING DATA")
    logger.info("=" * 80)
    
    # Load SMILES
    logger.info(f"Loading data from {CONFIG['data_dir']}")
    smiles_list, protein_ids_list = CPIDataLoader.load_dataset(CONFIG['data_dir'])
    
    # Build SMILES tokenizer
    logger.info("Building SMILES tokenizer vocabulary...")
    smiles_tokenizer = SMILESTokenizer()
    smiles_vocab = smiles_tokenizer.build_vocab(smiles_list, min_freq=1)
    
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(exist_ok=True)
    
    tokenizer_path = output_dir / 'smiles_tokenizer.pkl'
    smiles_tokenizer.save(str(tokenizer_path))
    logger.info(f"Saved SMILES tokenizer to {tokenizer_path}")
    logger.info(f"SMILES vocab size: {smiles_tokenizer.get_vocab_size()}")
    
    return smiles_tokenizer, smiles_list, protein_ids_list


def train_model(embeddings_file, smiles_tokenizer, smiles_list, protein_ids_list):
    """
    Stage 2: Train the model
    """
    logger.info("=" * 80)
    logger.info("STAGE 2: TRAINING MODEL")
    logger.info("=" * 80)
    
    device = CONFIG['device']
    logger.info(f"Using device: {device}")
    
    # Load protein embeddings
    logger.info(f"Loading protein embeddings from {embeddings_file}")
    protein_embeddings, embedding_protein_ids = CPIDataLoader.load_protein_embeddings(embeddings_file)
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader = CPIDataLoader.create_dataloaders(
        smiles_list=smiles_list,
        protein_ids_list=protein_ids_list,
        protein_embeddings=protein_embeddings,
        protein_embedding_ids=embedding_protein_ids,
        smiles_tokenizer=smiles_tokenizer,
        batch_size=CONFIG['batch_size'],
        train_split=CONFIG['train_split'],
        device=device,
        shuffle_train=True,
        num_workers=0
    )
    
    # Build Transformer model
    logger.info("Building Transformer model...")
    vocab_size = smiles_tokenizer.get_vocab_size()
    embedding_dim = protein_embeddings.shape[1]
    
    logger.info(f"SMILES vocab size: {vocab_size}")
    logger.info(f"Protein embedding dim: {embedding_dim}")
    
    transformer = build_transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        src_seq_len=CONFIG['max_protein_embedding_len'],
        tgt_seq_len=CONFIG['max_smiles_len'],
        d_model=CONFIG['d_model'],
        N=CONFIG['num_layers'],
        h=CONFIG['num_heads'],
        dropout=CONFIG['dropout'],
        d_ff=CONFIG['d_ff']
    )
    
    # Wrap in protein-drug model
    model = ProteinDrugTransformer(
        transformer=transformer,
        protein_embedding_dim=embedding_dim,
        protein_projection_dim=CONFIG['d_model']
    )
    
    # Training configuration
    train_config = TrainingConfig(
        batch_size=CONFIG['batch_size'],
        epochs=CONFIG['epochs'],
        learning_rate=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay'],
        warmup_steps=CONFIG['warmup_steps'],
        gradient_clip_val=CONFIG['gradient_clip_val'],
        label_smoothing=CONFIG['label_smoothing'],
        device=device
    )
    
    # Create trainer
    checkpoint_dir = Path(CONFIG['checkpoint_dir'])
    trainer = Trainer(
        model=model,
        config=train_config,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=str(checkpoint_dir)
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    logger.info("Training completed!")
    return model, smiles_tokenizer


def generate_drugs(model, smiles_tokenizer, protein_ids_list):
    """
    Stage 3: Generate drugs from proteins
    """
    logger.info("=" * 80)
    logger.info("STAGE 3: GENERATING DRUGS")
    logger.info("=" * 80)
    
    device = CONFIG['device']
    
    # Initialize generator
    generator = DrugGenerator(
        model=model,
        smiles_tokenizer=smiles_tokenizer,
        device=device,
        max_length=CONFIG['max_smiles_len']
    )
    
    # Load protein embeddings
    embeddings_file = Path(CONFIG['output_dir']) / CONFIG['protein_embeddings_file']
    protein_embeddings, embedding_protein_ids = CPIDataLoader.load_protein_embeddings(str(embeddings_file))
    
    # Select some proteins for generation
    num_generate = min(10, len(protein_embeddings))
    selected_indices = np.random.choice(len(protein_embeddings), size=num_generate, replace=False)
    
    logger.info(f"Generating drugs for {num_generate} proteins...")
    
    results = []
    validator = MolecularValidator()
    
    for idx in selected_indices:
        protein_id = embedding_protein_ids[idx]
        protein_emb = protein_embeddings[idx:idx+1]
        
        # Generate using different methods
        if CONFIG['generation_method'] == 'beam_search':
            predictions = generator.beam_search_decode(
                torch.FloatTensor(protein_emb),
                beam_width=CONFIG['beam_width']
            )
            # Get top prediction
            smiles = predictions[0][0] if predictions else ""
            score = predictions[0][1] if predictions else 0.0
        else:
            smiles = generator.greedy_decode(torch.FloatTensor(protein_emb))
            score = 0.0
        
        # Validate and canonicalize
        is_valid, message = validator.validate_smiles(smiles)
        canonical_smiles = validator.canonicalize_smiles(smiles)
        properties = validator.calculate_properties(canonical_smiles)
        
        result = {
            'protein_id': protein_id,
            'generated_smiles': smiles,
            'canonical_smiles': canonical_smiles,
            'is_valid': is_valid,
            'validation_message': message,
            'score': float(score),
            'properties': properties
        }
        results.append(result)
        
        logger.info(f"Protein {protein_id}:")
        logger.info(f"  Generated: {smiles}")
        logger.info(f"  Canonical: {canonical_smiles}")
        logger.info(f"  Valid: {is_valid}")
        logger.info(f"  Score: {score:.4f}")
    
    # Save results
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(exist_ok=True)
    results_file = output_dir / 'generation_results.json'
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Generation results saved to {results_file}")
    
    # Summary
    valid_count = sum(1 for r in results if r['is_valid'])
    logger.info(f"\nGeneration Summary:")
    logger.info(f"Total generated: {len(results)}")
    logger.info(f"Valid SMILES: {valid_count}/{len(results)}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Protein-to-Drug generation training pipeline")
    parser.add_argument('--stage', choices=['preprocess', 'train', 'generate', 'all'],
                       default='all', help='Stage to run')
    parser.add_argument('--data-dir', type=str, default=CONFIG['data_dir'],
                       help='Path to CPI data directory')
    parser.add_argument('--output-dir', type=str, default=CONFIG['output_dir'],
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default=CONFIG['device'],
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--epochs', type=int, default=CONFIG['epochs'],
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=CONFIG['batch_size'],
                       help='Batch size')
    
    args = parser.parse_args()
    
    # Update config
    CONFIG['data_dir'] = args.data_dir
    CONFIG['output_dir'] = args.output_dir
    CONFIG['device'] = args.device
    CONFIG['epochs'] = args.epochs
    CONFIG['batch_size'] = args.batch_size
    
    logger.info(f"Configuration: {json.dumps(CONFIG, indent=2)}")
    
    try:
        if args.stage in ['preprocess', 'all']:
            embeddings_file, smiles_list, protein_ids_list = preprocess_proteins()
        else:
            embeddings_file = str(Path(CONFIG['output_dir']) / CONFIG['protein_embeddings_file'])
            smiles_list, protein_ids_list = CPIDataLoader.load_dataset(CONFIG['data_dir'])
        
        if args.stage in ['train', 'all']:
            smiles_tokenizer, smiles_list, protein_ids_list = prepare_data(embeddings_file)
            model, smiles_tokenizer = train_model(embeddings_file, smiles_tokenizer, smiles_list, protein_ids_list)
        else:
            smiles_tokenizer, _, _ = prepare_data(embeddings_file)
            # Load best model
            checkpoint_path = Path(CONFIG['checkpoint_dir']) / 'best_model.pt'
            if not checkpoint_path.exists():
                logger.error(f"Model checkpoint not found: {checkpoint_path}")
                return
            # Would need to reconstruct model here for full implementation
        
        if args.stage in ['generate', 'all']:
            if args.stage == 'generate' and 'model' not in locals():
                # Load saved model
                logger.error("Model not trained. Run with --stage train first.")
                return
            generate_drugs(model, smiles_tokenizer, protein_ids_list)
        
        logger.info("Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in pipeline: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()