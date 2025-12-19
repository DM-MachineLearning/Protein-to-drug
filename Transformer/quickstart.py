"""
Quick Start Guide for Protein-to-Drug Generation

This script demonstrates how to use the pipeline programmatically.
"""

import sys
from pathlib import Path
import numpy as np
import torch

# Import pipeline components
from protein_encoder import ProteinEncoder, ProteinDatasetBuilder
from tokenizer import SMILESTokenizer
from data_loader import CPIDataLoader
from model import build_transformer
from train import ProteinDrugTransformer, TrainingConfig, Trainer
from inference import DrugGenerator, MolecularValidator


def example_1_protein_encoding():
    """Example 1: Encode proteins from UniProt"""
    print("\n" + "="*60)
    print("Example 1: Protein Encoding")
    print("="*60)
    
    # Initialize encoder
    encoder = ProteinEncoder(model_name="protbert")
    
    # Test sequence
    test_sequence = "MKFLKFSLLTAVLLSVVFAFSSCGDDDDTGFIFAAA"
    embedding = encoder.encode_sequence(test_sequence)
    
    print(f"Sequence: {test_sequence}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding (first 10 dims): {embedding[:10]}")
    
    # Fetch from UniProt
    print("\nFetching protein from UniProt...")
    uniprot_id = "P69905"  # Hemoglobin alpha
    sequence = encoder.fetch_sequence_from_uniprot(uniprot_id)
    if sequence:
        print(f"UniProt ID: {uniprot_id}")
        print(f"Sequence length: {len(sequence)}")
        print(f"First 50 amino acids: {sequence[:50]}")


def example_2_smiles_tokenization():
    """Example 2: SMILES Tokenization"""
    print("\n" + "="*60)
    print("Example 2: SMILES Tokenization")
    print("="*60)
    
    # Sample SMILES
    smiles_list = [
        "CCO",  # Ethanol
        "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",  # Ibuprofen
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
    ]
    
    # Create tokenizer
    tokenizer = SMILESTokenizer()
    
    # Build vocabulary
    vocab = tokenizer.build_vocab(smiles_list)
    print(f"Vocabulary size: {len(vocab)}")
    
    # Test encoding/decoding
    for smiles in smiles_list:
        encoded = tokenizer.encode(smiles)
        decoded = tokenizer.decode(encoded)
        print(f"Original:  {smiles}")
        print(f"Encoded:   {encoded[:20]}...")  # Show first 20 tokens
        print(f"Decoded:   {decoded}")
        print()


def example_3_data_loading():
    """Example 3: Load CPI Dataset"""
    print("\n" + "="*60)
    print("Example 3: Data Loading")
    print("="*60)
    
    try:
        data_dir = "../CPI/CPI"
        smiles_list, protein_ids = CPIDataLoader.load_dataset(data_dir)
        
        print(f"Loaded {len(smiles_list)} SMILES")
        print(f"Loaded {len(protein_ids)} protein IDs")
        print(f"\nFirst 5 samples:")
        for i in range(min(5, len(smiles_list))):
            print(f"  {i+1}. Protein: {protein_ids[i]}, SMILES: {smiles_list[i][:40]}...")
    except FileNotFoundError:
        print("Data directory not found. Skipping this example.")


def example_4_generate_from_embedding():
    """Example 4: Generate SMILES from Protein Embedding"""
    print("\n" + "="*60)
    print("Example 4: Drug Generation from Protein")
    print("="*60)
    
    # Create mock model and tokenizer for demonstration
    print("Note: This example requires a trained model.")
    print("See main.py --stage train for full training pipeline.")
    
    # Create dummy components
    from train import ProteinDrugTransformer
    
    # Build transformer
    transformer = build_transformer(
        src_vocab_size=100,
        tgt_vocab_size=100,
        src_seq_len=1024,
        tgt_seq_len=512,
        d_model=256,  # Small for demo
        N=2,
        h=4,
    )
    
    model = ProteinDrugTransformer(
        transformer=transformer,
        protein_embedding_dim=768,
        protein_projection_dim=256
    )
    
    # Initialize generator
    smiles_tokenizer = SMILESTokenizer()
    smiles_tokenizer.build_vocab(["CCO", "CC(C)O", "CCN"])
    
    generator = DrugGenerator(
        model=model,
        smiles_tokenizer=smiles_tokenizer,
        device='cpu'
    )
    
    # Generate from dummy protein embedding
    protein_embedding = torch.randn(1, 768)
    
    # Note: This will generate random output since model is untrained
    print("Generating SMILES from protein embedding...")
    print("(Note: Using random model - train for real results)")
    
    # Use sampling method
    smiles_samples = generator.sample_decode(
        protein_embedding,
        temperature=1.0,
        num_samples=3
    )
    
    print(f"Generated SMILES: {smiles_samples}")


def example_5_molecular_validation():
    """Example 5: Validate Molecular Properties"""
    print("\n" + "="*60)
    print("Example 5: Molecular Validation")
    print("="*60)
    
    validator = MolecularValidator()
    
    test_smiles = [
        "CCO",
        "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",
        "INVALID_SMILES",
    ]
    
    for smiles in test_smiles:
        is_valid, message = validator.validate_smiles(smiles)
        print(f"\nSMILES: {smiles}")
        print(f"Valid: {is_valid} ({message})")
        
        if is_valid:
            canonical = validator.canonicalize_smiles(smiles)
            properties = validator.calculate_properties(canonical)
            print(f"Canonical: {canonical}")
            if properties:
                print(f"Properties:")
                for prop, value in properties.items():
                    print(f"  {prop}: {value:.2f}" if isinstance(value, float) else f"  {prop}: {value}")


def example_6_full_pipeline():
    """Example 6: Full Pipeline"""
    print("\n" + "="*60)
    print("Example 6: Full Pipeline")
    print("="*60)
    
    print("""
    To run the full pipeline:
    
    1. PREPROCESS PROTEINS:
       python main.py --stage preprocess
       - Fetches protein sequences from UniProt
       - Encodes them using ProtBERT
       - Saves embeddings to results/protein_embeddings.npz
    
    2. TRAIN MODEL:
       python main.py --stage train --epochs 50
       - Builds SMILES tokenizer
       - Creates dataloaders
       - Trains Transformer model
       - Saves checkpoints
    
    3. GENERATE DRUGS:
       python main.py --stage generate
       - Loads trained model
       - Generates SMILES for proteins
       - Validates output
       - Saves results
    
    4. RUN ALL STAGES:
       python main.py --stage all --epochs 50
    """)


def main():
    print("\n" + "="*60)
    print("Protein-to-Drug Generation - Quick Start Examples")
    print("="*60)
    
    examples = [
        ("Protein Encoding", example_1_protein_encoding),
        ("SMILES Tokenization", example_2_smiles_tokenization),
        ("Data Loading", example_3_data_loading),
        ("Drug Generation", example_4_generate_from_embedding),
        ("Molecular Validation", example_5_molecular_validation),
        ("Full Pipeline", example_6_full_pipeline),
    ]
    
    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    
    print("\nRunning all examples...\n")
    
    for name, func in examples:
        try:
            func()
        except Exception as e:
            print(f"\n⚠ Error in {name}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
    
    print("\n" + "="*60)
    print("Examples complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Review the README_PROTEIN_DRUG.md for detailed documentation")
    print("2. Prepare your data or use the provided CPI dataset")
    print("3. Run: python main.py --stage all")
    print("\nFor more information, see:")
    print("- protein_encoder.py: Protein encoding")
    print("- tokenizer.py: SMILES tokenization")
    print("- data_loader.py: Data loading")
    print("- train.py: Model training")
    print("- inference.py: Drug generation")
    print("- main.py: Full pipeline orchestration")
    print()
