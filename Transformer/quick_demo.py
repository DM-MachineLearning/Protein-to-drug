"""
Quick demo: Train for 3 epochs and generate samples (no UniProt API calls)
"""

import torch
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm

# Set device
device = torch.device("cpu")

print("\n" + "="*80)
print("QUICK DEMO: Training & Generation")
print("="*80 + "\n")

# ============================================================================
# STEP 1: Create mock data (skip UniProt preprocessing)
# ============================================================================
print("STEP 1: Creating mock protein embeddings...")

from tokenizer import SMILESTokenizer, ProteinTokenizer
from data_loader import CPIDataLoader, CPIDataset
from model import build_transformer
from train import ProteinDrugTransformer, Trainer, TrainingConfig

# Create sample SMILES and embeddings
sample_smiles = [
    "CC(C)Cc1ccc(cc1)C(C)C(O)=O",
    "CC(=O)Oc1ccccc1C(=O)O",
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "Cc1oncc1C(=O)Nc2ccc(cc2)C(F)(F)F",
    "O=C(O)Cc1ccccc1Nc2c(Cl)cccc2Cl",
    "CC(C)CC(NC(=O)c1ccc(Cl)cc1)C(=O)N[C@@H](Cc2ccccc2)C(=O)N",
    "c1ccc2c(c1)ccc3c2cccc3",
    "c1cc(Cl)ccc1Cl",
]

num_proteins = 32  # Match the number of SMILES we'll create
protein_embedding_dim = 768
max_samples = min(50, len(sample_smiles) * 5)

# Create mock protein embeddings
protein_embeddings = np.random.randn(num_proteins, protein_embedding_dim).astype(np.float32)

print(f"Sample SMILES available: {len(sample_smiles)}")

# Create mock dataset file
output_dir = Path("./results")
output_dir.mkdir(exist_ok=True)
embeddings_file = output_dir / "protein_embeddings_demo.npz"

np.savez(
    embeddings_file,
    embeddings=protein_embeddings,
    protein_ids=np.arange(num_proteins)
)

print(f"✓ Created {num_proteins} mock protein embeddings: {embeddings_file}")

# ============================================================================
# STEP 2: Build tokenizers
# ============================================================================
print("\nSTEP 2: Building tokenizers...")

smiles_tokenizer = SMILESTokenizer()
smiles_tokenizer.build_vocab(sample_smiles)

protein_tokenizer = ProteinTokenizer()

print(f"✓ SMILES vocab size: {len(smiles_tokenizer.vocab)}")
print(f"✓ Protein vocab size: {len(protein_tokenizer.vocab)}")

# ============================================================================
# STEP 3: Create training dataset
# ============================================================================
print("\nSTEP 3: Creating training dataset...")

# Create simple pairs
train_data = []
for i in range(min(32, len(sample_smiles) * 4)):
    protein_idx = i % num_proteins
    smiles_idx = i % len(sample_smiles)
    protein_id = f"P{protein_idx:05d}"  # Must match embedding IDs
    train_data.append({
        "protein_id": protein_id,
        "smiles": sample_smiles[smiles_idx]
    })

# Save to files for CPIDataLoader
data_dir = Path("./results/demo_data")
data_dir.mkdir(exist_ok=True)

with open(data_dir / "smiles.smi", "w") as f:
    for item in train_data:
        f.write(item["smiles"] + "\n")

with open(data_dir / "uniprot_ID.smi", "w") as f:
    for item in train_data:
        # Extract numeric ID from protein_id
        idx = int(item["protein_id"][1:])
        f.write(str(idx) + "\n")

print(f"✓ Created dataset with {len(train_data)} samples")

# ============================================================================
# STEP 4: Create dataloaders
# ============================================================================
print("\nSTEP 4: Creating dataloaders...")

# Load data
smiles_list, protein_ids_list = CPIDataLoader.load_dataset(str(data_dir))
protein_embeddings_loaded, protein_ids_emb = CPIDataLoader.load_protein_embeddings(str(embeddings_file))

# Manually create dataset (bypass ID matching)
dataset = CPIDataset(
    protein_embeddings=protein_embeddings_loaded,
    protein_ids=[f"P{i:05d}" for i in range(len(protein_embeddings_loaded))],
    smiles_list=smiles_list,
    smiles_tokenizer=smiles_tokenizer,
    protein_tokenizer=protein_tokenizer,
    max_smiles_len=512,
    device=device
)

# Split into train/val
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False)

print(f"✓ Train batches: {len(train_loader)}")
print(f"✓ Val batches: {len(val_loader)}")

# ============================================================================
# STEP 5: Create model
# ============================================================================
print("\nSTEP 5: Creating model...")

d_model = 256
vocab_size = len(smiles_tokenizer.vocab)

# Build transformer
transformer = build_transformer(
    src_vocab_size=vocab_size,
    tgt_vocab_size=vocab_size,
    src_seq_len=512,
    tgt_seq_len=512,
    d_model=d_model,
    N=2,  # num layers
    h=4,  # num heads
    dropout=0.1,
    d_ff=512
)

transformer = transformer.to(device)

model = ProteinDrugTransformer(
    transformer=transformer,
    protein_embedding_dim=protein_embedding_dim,
    protein_projection_dim=256
).to(device)

print(f"✓ Model created")
print(f"  - Transformer d_model: 256")
print(f"  - Num layers: 2")
print(f"  - Num heads: 4")

# ============================================================================
# STEP 6: Train for 3 epochs
# ============================================================================
print("\nSTEP 6: Training for 3 epochs...")
print("-" * 80)

config = TrainingConfig(
    epochs=3,
    batch_size=4,
    learning_rate=1e-3,
    warmup_steps=100,
    device=device
)

trainer = Trainer(
    model=model,
    config=config,
    train_loader=train_loader,
    val_loader=val_loader,
    output_dir="./checkpoints/demo"
)

history = trainer.train()

print("-" * 80)
print(f"✓ Training completed!")
print(f"  - Final train loss: {trainer.training_history['train_loss'][-1]:.4f}")
print(f"  - Final val loss: {trainer.training_history['val_loss'][-1]:.4f}")

# ============================================================================
# STEP 7: Generate samples
# ============================================================================
print("\nSTEP 7: Generating drug samples...")
print("-" * 80)

from inference import DrugGenerator, MolecularValidator

generator = DrugGenerator(model, smiles_tokenizer, device=device)
validator = MolecularValidator()

# Get first batch of protein embeddings
sample_proteins = torch.tensor(protein_embeddings[:3], dtype=torch.float32).to(device)

# Generate SMILES
print("\nGenerating SMILES (greedy decoding)...\n")

generated_smiles = []
for i in range(3):
    protein_emb = sample_proteins[i:i+1]
    
    # Generate multiple SMILES
    for method, method_name in [
        ("greedy", "Greedy"),
        ("beam_search", "Beam Search"),
        ("sample", "Sampling")
    ]:
        try:
            smiles = generator.generate(
                protein_emb,
                method=method,
                max_length=100,
                temperature=1.0
            )
            
            # Validate
            is_valid = validator.validate_smiles(smiles)
            props = validator.calculate_properties(smiles) if is_valid else {}
            
            generated_smiles.append({
                "protein_idx": i,
                "method": method_name,
                "smiles": smiles,
                "valid": is_valid,
                "properties": props
            })
            
            status = "✓ VALID" if is_valid else "✗ INVALID"
            print(f"  [{method_name}] {status}: {smiles[:60]}")
            
        except Exception as e:
            print(f"  [{method_name}] ERROR: {str(e)[:50]}")
            generated_smiles.append({
                "protein_idx": i,
                "method": method_name,
                "smiles": None,
                "valid": False,
                "error": str(e)
            })

# ============================================================================
# STEP 8: Show results
# ============================================================================
print("\n" + "="*80)
print("RESULTS SUMMARY")
print("="*80 + "\n")

# Training results
print("TRAINING METRICS:")
print(f"  Epochs: 3")
print(f"  Starting train loss: {history['train_loss'][0]:.4f}")
print(f"  Final train loss: {history['train_loss'][-1]:.4f}")
print(f"  Loss improvement: {history['train_loss'][0] - history['train_loss'][-1]:.4f}")
print(f"  Final val loss: {history['val_loss'][-1]:.4f}")

# Generation results
print("\nGENERATION RESULTS:")
valid_count = sum(1 for s in generated_smiles if s["valid"])
total_count = len(generated_smiles)
print(f"  Generated samples: {total_count}")
print(f"  Valid SMILES: {valid_count}/{total_count} ({100*valid_count/total_count:.1f}%)")

# Sample outputs
print("\nSAMPLE OUTPUTS:")
for item in generated_smiles[:6]:
    if item["valid"]:
        props = item.get("properties", {})
        print(f"\n  [{item['method']}] Protein {item['protein_idx']}")
        print(f"    SMILES: {item['smiles'][:70]}")
        if props:
            print(f"    MW: {props.get('MW', 'N/A'):.1f} | LogP: {props.get('LogP', 'N/A'):.2f}")

# Save results
results_file = output_dir / "demo_results.json"
results_summary = {
    "training": {
        "epochs": 3,
        "initial_train_loss": float(history['train_loss'][0]),
        "final_train_loss": float(history['train_loss'][-1]),
        "final_val_loss": float(history['val_loss'][-1]),
    },
    "generation": {
        "total_samples": total_count,
        "valid_samples": valid_count,
        "valid_percentage": 100 * valid_count / total_count,
    },
    "samples": generated_smiles[:6]  # First 6 samples
}

with open(results_file, "w") as f:
    json.dump(results_summary, f, indent=2, default=str)

print(f"\n✓ Results saved to: {results_file}")

print("\n" + "="*80)
print("DEMO COMPLETE!")
print("="*80 + "\n")

print("Summary:")
print(f"  ✓ Created mock dataset")
print(f"  ✓ Built tokenizers")
print(f"  ✓ Trained model for 3 epochs")
print(f"  ✓ Generated {total_count} drug samples")
print(f"  ✓ Validated chemistry")
print(f"  ✓ Saved results")
