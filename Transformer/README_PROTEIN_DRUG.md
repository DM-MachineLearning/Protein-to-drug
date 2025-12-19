# Protein-to-Drug Generation using Transformer

A complete pipeline for generating drug compounds (SMILES) from protein targets using a Transformer architecture. This system learns to map protein characteristics to desired molecular structures.

## Overview

The pipeline consists of three main stages:

1. **Protein Preprocessing**: Fetch protein sequences from UniProt and encode them using pretrained protein language models
2. **Model Training**: Train a Transformer model to generate SMILES from protein embeddings
3. **Drug Generation**: Generate novel drug compounds for target proteins

## Architecture

### Components

- **Protein Encoder** (`protein_encoder.py`): 
  - Supports multiple protein language models (ProtBERT, ESM2, ProtTrans)
  - Fetches sequences from UniProt API
  - Caches sequences and embeddings for efficiency
  - Falls back to one-hot encoding if models unavailable

- **SMILES Tokenizer** (`tokenizer.py`):
  - Regex-based tokenization of SMILES strings
  - Builds vocabulary from training data
  - Special tokens: `<PAD>`, `<SOS>`, `<EOS>`, `<UNK>`

- **Data Loader** (`data_loader.py`):
  - Loads CPI (Compound-Protein Interaction) dataset
  - Handles protein-SMILES pairing
  - Creates train/val split with proper batching

- **Transformer Model** (`model.py`, `train.py`):
  - Encoder-Decoder architecture
  - Multi-head attention with positional encoding
  - Protein embeddings as encoder input
  - SMILES generation as decoder output

- **Training** (`train.py`):
  - Cross-entropy loss with label smoothing
  - AdamW optimizer with learning rate warmup
  - Gradient clipping and checkpointing

- **Inference** (`inference.py`):
  - Greedy, beam search, and sampling-based generation
  - Molecular validity checking via RDKit
  - Property calculation (MW, LogP, TPSA, etc.)

## Installation

### 1. Install Dependencies

```bash
cd Transformer
pip install -r requirements.txt
```

### 2. Optional: Install Protein Language Models

For better protein embeddings, install optional models:

```bash
# For ESM2 (Meta)
pip install fair-esm[esmfold]

# For ProtTrans (already included via transformers)
# T5 models will auto-download on first use
```

## Usage

### Quick Start

Run the complete pipeline:

```bash
python main.py --stage all --epochs 50 --batch-size 32
```

### By Stages

**Stage 1: Preprocess Proteins**
```bash
python main.py --stage preprocess
```
- Loads SMILES and protein IDs from CPI dataset
- Fetches protein sequences from UniProt
- Encodes proteins to embeddings (cached)
- Saves embeddings to `results/protein_embeddings.npz`

**Stage 2: Train Model**
```bash
python main.py --stage train --epochs 50
```
- Builds SMILES tokenizer vocabulary
- Creates train/val dataloaders
- Trains Transformer model
- Saves checkpoints to `checkpoints/`

**Stage 3: Generate Drugs**
```bash
python main.py --stage generate
```
- Loads trained model
- Generates SMILES for sample proteins
- Validates and canonicalizes output
- Saves results to `results/generation_results.json`

### Command Line Options

```bash
python main.py \
  --stage [preprocess|train|generate|all]  # Pipeline stage
  --data-dir ../CPI/CPI                     # Path to data
  --output-dir ./results                    # Output directory
  --device cuda                             # Device (cuda/cpu)
  --epochs 50                               # Training epochs
  --batch-size 32                           # Batch size
```

## Data Format

Expected directory structure:
```
CPI/CPI/
├── smiles.smi          # One SMILES per line (551k+ compounds)
└── uniprot_ID.smi      # One UniProt ID per line (matching order)
```

Example:
```
# smiles.smi
CCO
CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O
...

# uniprot_ID.smi
P12345
P12345
...
```

## Configuration

Edit `CONFIG` in `main.py` to customize:

```python
CONFIG = {
    # Data
    'data_dir': '../CPI/CPI',
    
    # Model
    'd_model': 512,           # Transformer dimension
    'num_layers': 6,          # Number of encoder/decoder layers
    'num_heads': 8,           # Multi-head attention heads
    'd_ff': 2048,             # Feed-forward dimension
    'dropout': 0.1,
    
    # Training
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 3e-4,
    'warmup_steps': 4000,
    
    # Protein encoding
    'protein_model': 'protbert',  # 'esm2', 'prottrans'
    
    # Generation
    'generation_method': 'beam_search',
    'beam_width': 5,
}
```

## Model Architecture

### Protein Encoding
- Input: UniProt ID → Fetch sequence → Encode with language model
- Output: Fixed-size embedding (e.g., 1024-dim for ProtBERT)
- Linear projection to `d_model` dimension

### Transformer
```
Input: Protein Embedding
  ↓
Encoder: Single protein embedding token
  ↓
Decoder: Generate SMILES tokens autoregressively
  ↓
Output: Logits over SMILES vocabulary
```

### Generation Methods

1. **Greedy**: Select highest probability token at each step
2. **Beam Search**: Explore multiple hypothesis sequences, pick best
3. **Sampling**: Temperature-based sampling with top-k/top-p filtering

## Output Files

After running the pipeline:

```
results/
├── protein_embeddings.npz           # Cached protein embeddings
├── protein_sequences_cache.npz      # Cached sequences
├── smiles_tokenizer.pkl             # Saved tokenizer
└── generation_results.json          # Generation output

checkpoints/
├── best_model.pt                    # Best validation checkpoint
├── checkpoint_epoch_*.pt            # Periodic checkpoints
└── training_history.json            # Loss curves
```

## Example Results

```json
[
  {
    "protein_id": "P12345",
    "generated_smiles": "CCO",
    "canonical_smiles": "CCO",
    "is_valid": true,
    "score": 0.95,
    "properties": {
      "molecular_weight": 46.04,
      "logp": -0.27,
      "num_h_donors": 1,
      "num_h_acceptors": 1,
      "num_rotatable_bonds": 0,
      "tpsa": 20.23
    }
  }
]
```

## Performance

Expected metrics:
- **Valid SMILES**: 70-85% (depends on model training)
- **Training Time**: ~2-4 hours on GPU (50 epochs, 32GB GPU)
- **Inference Speed**: ~100-500 SMILES/second per GPU

## Troubleshooting

### Out of Memory
- Reduce `batch_size`
- Reduce `d_model` or `num_layers`
- Use gradient accumulation

### Poor Generation Quality
- Train for more epochs
- Increase `d_model` to 768 or 1024
- Use stronger protein encoder (ESM2 > ProtBERT > simple encoding)
- Adjust `learning_rate` and `warmup_steps`

### UniProt API Timeout
- Check internet connection
- Sequences are cached, later runs will be faster
- Can manually prepare sequence file

## References

- [Transformer Architecture](https://arxiv.org/abs/1706.03762)
- [ProtBERT](https://github.com/agemagician/ProtTrans)
- [ESM2](https://github.com/facebookresearch/esm)
- [SMILES Format](https://en.wikipedia.org/wiki/Simplified_molecular_input_line_entry_system)
- [Compound-Protein Interaction](https://en.wikipedia.org/wiki/Protein%E2%80%93ligand_interaction)

## License

MIT License

## Future Improvements

- [ ] Multi-task learning (protein, drug, interaction)
- [ ] Conditional generation (e.g., target molecular weight)
- [ ] Graph neural networks for SMILES
- [ ] Reinforcement learning for property optimization
- [ ] Molecular docking validation
- [ ] Active learning for targeted discovery
