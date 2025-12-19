"""
ARCHITECTURE DOCUMENTATION

Protein-to-Drug Generation System Architecture
"""

"""
╔════════════════════════════════════════════════════════════════════════════════╗
║                    PROTEIN-TO-DRUG GENERATION ARCHITECTURE                     ║
╚════════════════════════════════════════════════════════════════════════════════╝


┌─────────────────────────────────────────────────────────────────────────────────┐
│                              DATA PIPELINE FLOW                                  │
└─────────────────────────────────────────────────────────────────────────────────┘

INPUT DATA (CPI DATASET)
    │
    ├── smiles.smi ──────────────┬──────────────────────┐
    │   (SMILES strings)         │                      │
    │   551,224 compounds        │                      │
    │                            │                      ▼
    └── uniprot_ID.smi           │          ┌──────────────────────┐
        (Protein IDs)            │          │  SMILES TOKENIZER    │
        551,224 proteins         │          │  (tokenizer.py)      │
                                 │          ├──────────────────────┤
                                 │          │ • Regex tokenization │
                                 │          │ • Vocab building     │
                                 │          │ • Encode/decode      │
                                 │          │ • 70-100 tokens      │
                                 │          └──────────────────────┘
                                 │                   │
                                 │                   ▼
    PROTEIN FETCHING             │         ┌──────────────────────┐
    ├─ UniProt API               │         │  SMILES EMBEDDINGS   │
    │  └─ Fetch sequences        │         │  (Token indices)     │
    │     (cached)               │         └──────────────────────┘
    │                            │
    ├─ Unique proteins: ~70k     │
    │                            │
    ▼                            ▼
┌──────────────────┐    ┌──────────────────────┐
│ PROTEIN ENCODER  │    │  TRAINING DATASET    │
│ (protein_encoder.│    │  (data_loader.py)    │
│ py)              │    ├──────────────────────┤
├──────────────────┤    │ • Protein-SMILES     │
│ Models:          │    │   pairs              │
│ • ProtBERT       │    │ • Batching           │
│ • ESM2           │    │ • Padding/masking    │
│ • ProtTrans      │    │ • Train/val split    │
│ • One-hot (FB)   │    │ • 80/20 split        │
├──────────────────┤    └──────────────────────┘
│ Output:          │           │
│ • 768-1024 dims  │           ▼
│ • Embeddings     │    ┌──────────────────────┐
│ • Cached (.npz)  │    │  PYTORCH DATALOADER  │
└──────────────────┘    │  (PyTorch built-in)  │
                        ├──────────────────────┤
                        │ Batch size: 32       │
                        │ Workers: 0-4         │
                        │ Pin memory: true     │
                        └──────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────────┐
│                           MODEL ARCHITECTURE                                     │
└─────────────────────────────────────────────────────────────────────────────────┘

INPUT LAYER
    │
    ├─ Protein Embedding (1, 768)
    │       │
    │       ▼
    │  ┌──────────────────────────────┐
    │  │ Linear Projection            │
    │  │ (768 → 512)                  │
    │  │ protein_projection.py         │
    │  └──────────────────────────────┘
    │       │
    │       ▼ (1, 512)
    │  ┌──────────────────────────────┐
    │  │ Unsqueeze to token           │
    │  │ (1, 1, 512)                  │
    │  │ = single token encoder input │
    │  └──────────────────────────────┘
    │       │
    │       ▼
    │  ┌─────────────────────────────────────────────────────────┐
    │  │              ENCODER (Single Token)                     │
    │  │  (from model.py - build_transformer)                   │
    │  ├─────────────────────────────────────────────────────────┤
    │  │ • Input Embedding: (1, 1, 512)                          │
    │  │ • Positional Encoding: adds position info              │
    │  │ • 6 Encoder Layers:                                     │
    │  │   ├─ Self-Attention (8 heads)                           │
    │  │   │  └─ Attend to self (1 token only)                  │
    │  │   └─ Feed-Forward (512 → 2048 → 512)                   │
    │  │     └─ ReLU activation                                  │
    │  │ • Layer Normalization (each layer)                      │
    │  │ • Dropout (0.1)                                         │
    │  └─────────────────────────────────────────────────────────┘
    │       │
    │       ▼ encoder_output (1, 1, 512)
    │
    ├─ SMILES Sequence
    │  (batch_size, seq_len) ─── tokenized SMILES
    │       │
    │       ▼
    │  ┌──────────────────────────────┐
    │  │ Token Embedding              │
    │  │ (vocab_size × 512)           │
    │  │ SMILESTokenizer              │
    │  └──────────────────────────────┘
    │       │
    │       ▼ (batch, seq_len, 512)
    │  ┌──────────────────────────────┐
    │  │ Positional Encoding          │
    │  │ (relative positions)         │
    │  └──────────────────────────────┘
    │       │
    │       ▼
    │  ┌─────────────────────────────────────────────────────────┐
    │  │             DECODER (Autoregressive)                    │
    │  │  (from model.py)                                        │
    │  ├─────────────────────────────────────────────────────────┤
    │  │ • Input: Decoder Embedding (batch, seq_len, 512)       │
    │  │ • 6 Decoder Layers:                                     │
    │  │   ├─ Self-Attention (8 heads, causal mask)             │
    │  │   │  └─ Can only attend to earlier tokens              │
    │  │   ├─ Cross-Attention (8 heads)                          │
    │  │   │  └─ Attends to encoder output (1 token)            │
    │  │   └─ Feed-Forward (512 → 2048 → 512)                   │
    │  │ • Layer Normalization & Dropout                         │
    │  └─────────────────────────────────────────────────────────┘
    │       │
    │       ▼ decoder_output (batch, seq_len, 512)
    │
    ▼
PROJECTION LAYER
    │
    ▼
┌──────────────────────────────────────────────┐
│ Linear Projection                            │
│ (512 → vocab_size)                           │
│ ProjectionLayer (model.py)                   │
├──────────────────────────────────────────────┤
│ • Maps each token to vocabulary scores       │
│ • Output: (batch, seq_len, vocab_size)      │
│ • Log-softmax activation                    │
└──────────────────────────────────────────────┘
    │
    ▼
OUTPUT
    Logits for each token position
    Shape: (batch_size, max_smiles_len, vocab_size)


┌─────────────────────────────────────────────────────────────────────────────────┐
│                          TRAINING PIPELINE                                       │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐
│  BATCH INPUT    │
├─────────────────┤
│ protein_emb: (32, 768)
│ decoder_input: (32, 512)
│ labels: (32, 512)
│ masks: (32, 512)
└─────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  FORWARD PASS                               │
│  (ProteinDrugTransformer.forward)          │
├─────────────────────────────────────────────┤
│ 1. Project protein embedding                │
│ 2. Pass through encoder                     │
│ 3. Pass decoder input through decoder       │
│ 4. Project to vocabulary                    │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  LOGITS OUTPUT                              │
│  Shape: (32, 512, vocab_size)              │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  LOSS CALCULATION                           │
│  CrossEntropyLoss (train.py)               │
├─────────────────────────────────────────────┤
│ • Reshape: (batch * seq_len, vocab_size)   │
│ • Label smoothing: 0.1                      │
│ • Ignore padding tokens (index 0)           │
│ • Compute cross-entropy loss                │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  BACKPROPAGATION                            │
│  (optimizer.step)                           │
├─────────────────────────────────────────────┤
│ • Zero gradients                            │
│ • Backward pass                             │
│ • Clip gradients (max 1.0)                  │
│ • AdamW optimizer step                      │
│ • Learning rate scheduler step              │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  CHECKPOINT SAVING                          │
│  (Trainer.save_checkpoint)                  │
├─────────────────────────────────────────────┤
│ • Save model state                          │
│ • Save optimizer state                      │
│ • Save scheduler state                      │
│ • Keep best model                           │
│ • Track training history                    │
└─────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────────┐
│                        GENERATION/INFERENCE                                      │
└─────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────┐
│ PROTEIN EMBEDDING│
│ (1, 768)         │
└──────────────────┘
    │
    ▼
┌──────────────────────────────────────────┐
│  GENERATION METHOD SELECTION             │
│  (DrugGenerator - inference.py)          │
└──────────────────────────────────────────┘
    │
    ├─ GREEDY (Fast)
    │  ├─ For each position:
    │  │   ├─ Run model
    │  │   ├─ Select argmax token
    │  │   └─ Add to sequence
    │  └─ ~1-2ms per SMILES
    │
    ├─ BEAM SEARCH (Quality) ◄── Default
    │  ├─ Keep k=5 best candidates
    │  ├─ Expand each at each step
    │  ├─ Prune low-scoring sequences
    │  └─ ~5-10ms per SMILES
    │
    └─ SAMPLING (Diversity)
       ├─ Temperature scaling
       ├─ Top-k filtering
       ├─ Nucleus (top-p) sampling
       ├─ Multiple samples per protein
       └─ ~3-5ms per sample

    │
    ▼
┌──────────────────────────────────────────┐
│  GENERATED SMILES STRING                 │
│  e.g. "CCO", "CC(C)O", "CCCO"           │
└──────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────┐
│  VALIDATION (MolecularValidator)         │
│  (inference.py)                          │
├──────────────────────────────────────────┤
│ • RDKit SMILES parsing                   │
│ • Canonicalization                       │
│ • Property calculation:                  │
│   ├─ Molecular weight                    │
│   ├─ LogP (lipophilicity)                │
│   ├─ H-bond donors/acceptors             │
│   ├─ Rotatable bonds                     │
│   └─ TPSA (topological polar surface)    │
└──────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────┐
│  RESULTS (JSON)                          │
│  {                                       │
│    "protein_id": "P12345",               │
│    "generated_smiles": "CCO",            │
│    "canonical_smiles": "CCO",            │
│    "is_valid": true,                     │
│    "score": 0.95,                        │
│    "properties": {...}                   │
│  }                                       │
└──────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────────┐
│                          FILE STRUCTURE                                          │
└─────────────────────────────────────────────────────────────────────────────────┘

Transformer/
├── protein_encoder.py        ← Protein encoding + UniProt API
├── tokenizer.py              ← SMILES/Protein tokenization
├── data_loader.py            ← Dataset & dataloader
├── dataset.py                ← BilingualDataset (utilities)
├── model.py                  ← Transformer architecture
├── train.py                  ← Training loop & optimizer
├── inference.py              ← Generation & validation
├── main.py                   ← Pipeline orchestration
├── quickstart.py             ← Interactive examples
├── requirements.txt          ← Dependencies
├── setup.sh / setup.bat      ← Environment setup
├── README_PROTEIN_DRUG.md    ← User documentation
└── IMPLEMENTATION_SUMMARY.md ← This file


┌─────────────────────────────────────────────────────────────────────────────────┐
│                          KEY COMPONENTS                                          │
└─────────────────────────────────────────────────────────────────────────────────┘

CLASS HIERARCHY:

main.py (Orchestration)
    ├── preprocess_proteins()
    │   └── ProteinEncoder
    │       └── ProteinDatasetBuilder
    │
    ├── train_model()
    │   ├── CPIDataLoader
    │   ├── ProteinDrugTransformer
    │   │   └── build_transformer()
    │   └── Trainer
    │
    └── generate_drugs()
        ├── DrugGenerator
        │   └── [greedy|beam_search|sample]_decode()
        └── MolecularValidator
            └── [validate_smiles|canonicalize|calculate_properties]


┌─────────────────────────────────────────────────────────────────────────────────┐
│                        CONFIGURATION PARAMETERS                                  │
└─────────────────────────────────────────────────────────────────────────────────┘

MODEL:
  d_model: 512 ................... Transformer dimension
  num_layers: 6 .................. Encoder/decoder layers
  num_heads: 8 ................... Multi-head attention heads
  d_ff: 2048 ..................... Feed-forward inner dimension
  dropout: 0.1 ................... Dropout rate
  max_smiles_len: 512 ............ Max SMILES sequence length

TRAINING:
  batch_size: 32 ................. Batch size
  epochs: 50 ..................... Training epochs
  learning_rate: 3e-4 ............ Initial LR
  warmup_steps: 4000 ............. LR warmup steps
  gradient_clip_val: 1.0 ......... Gradient clipping
  label_smoothing: 0.1 ........... Label smoothing
  weight_decay: 1e-5 ............. L2 regularization

DATA:
  train_split: 0.8 ............... Train/val split

PROTEIN:
  protein_model: 'protbert' ....... Encoder model
  protein_embedding_dim: 768 ..... Protein embedding size

GENERATION:
  generation_method: 'beam_search' Decoding method
  beam_width: 5 .................. Beam search width
  temperature: 1.0 ............... Sampling temperature
  top_k: 0 ....................... Top-k sampling threshold
  top_p: 0.9 ..................... Nucleus sampling threshold
"""

if __name__ == "__main__":
    print(__doc__)
