"""
COMPLETE SETUP & USAGE GUIDE
Protein-to-Drug Generation with Transformer

Last Updated: December 2025
"""

# ============================================================================
# TABLE OF CONTENTS
# ============================================================================
"""
1. QUICK START (5 minutes)
2. DETAILED INSTALLATION
3. DATA PREPARATION
4. TRAINING THE MODEL
5. GENERATING DRUGS
6. ADVANCED USAGE
7. TROUBLESHOOTING
8. PERFORMANCE OPTIMIZATION
9. EXPECTED RESULTS
10. REFERENCES
"""

# ============================================================================
# 1. QUICK START (5 MINUTES)
# ============================================================================

"""
STEP 1: Install Dependencies
    cd Transformer
    pip install -r requirements.txt
    pip install fair-esm  # Optional but recommended

STEP 2: Run Setup Script
    Windows:  setup.bat
    Linux/Mac: bash setup.sh

STEP 3: Run Pipeline
    python main.py --stage all --epochs 50

That's it! The pipeline will:
    • Fetch proteins from UniProt
    • Encode them with ProtBERT
    • Train the Transformer model
    • Generate drugs for test proteins
    • Save results to results/generation_results.json
"""

# ============================================================================
# 2. DETAILED INSTALLATION
# ============================================================================

"""
PREREQUISITES:
    • Python 3.10 or higher
    • CUDA 11.8 (recommended for GPU) or CPU-only
    • 16GB RAM minimum (32GB recommended)
    • 10GB disk space for models and data

STEP-BY-STEP INSTALLATION:

1. Create Virtual Environment
   
   Windows:
   > python -m venv venv
   > venv\Scripts\activate.bat
   
   Linux/Mac:
   $ python -m venv venv
   $ source venv/bin/activate

2. Upgrade pip
   
   pip install --upgrade pip setuptools wheel

3. Install PyTorch
   
   GPU (CUDA 11.8):
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   CPU only:
   pip install torch torchvision torchaudio

4. Install Project Dependencies
   
   pip install -r requirements.txt

5. Optional: Install Protein Models
   
   For ESM2 (recommended for better quality):
   pip install fair-esm
   
   For ProtTrans:
   # Auto-downloaded on first use

6. Create Output Directories
   
   mkdir results checkpoints

7. Verify Installation
   
   python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
   python -c "import transformers; import rdkit; print('All packages imported successfully')"
"""

# ============================================================================
# 3. DATA PREPARATION
# ============================================================================

"""
YOUR DATA SHOULD BE IN:
    ../CPI/CPI/
    ├── smiles.smi       # SMILES strings (one per line)
    └── uniprot_ID.smi   # UniProt IDs (one per line, matching order)

EXPECTED DATA FORMAT:

smiles.smi (first 5 lines):
    [CH2]C1CC(n2ccc3c(-c4cc5cc(OC)ccc5[nH]4)ncnc32)CC1O
    [CH2]C1CC(n2ccc3c(-c4cc5ccc(C)cc5[nH]4)ncnc32)CC1O
    [CH2]C1CC(n2ccc3c(-c4cc5ccccc5[nH]4)nc(C)nc32)CC1O
    NS(=O)(=O)OCC1CC(n2ccc3c(-c4cc5ccccc5[nH]4)ncnc32)CC1O
    [CH2]C1CC(n2ccc3c(-c4cc5ccc(Cl)cc5[nH]4)ncnc32)CC1O

uniprot_ID.smi (first 5 lines):
    A0AVT1
    A0AVT1
    A0AVT1
    A0AVT1
    A0AVT1

DATA VALIDATION:
    • SMILES strings must be valid (parseable by RDKit)
    • UniProt IDs should exist in UniProt database
    • Equal number of lines in both files
    • No empty lines
    • One entry per line

OPTIONAL: Use Your Own Data
    1. Format your data as above
    2. Place in appropriate location
    3. Update data_dir in main.py CONFIG
    4. Run: python main.py --stage preprocess
"""

# ============================================================================
# 4. TRAINING THE MODEL
# ============================================================================

"""
SINGLE-STAGE TRAINING:

1. Preprocess Proteins (fetch + encode)
   
   python main.py --stage preprocess
   
   Output:
   • results/protein_embeddings.npz (embeddings + IDs)
   • results/protein_sequences_cache.npz (cached sequences)
   • Logs progress for each protein

2. Train Model
   
   python main.py --stage train --epochs 50 --batch-size 32
   
   Output:
   • checkpoints/best_model.pt (best model)
   • checkpoints/checkpoint_epoch_*.pt (periodic checkpoints)
   • checkpoints/training_history.json (metrics)
   • Console logging of losses

3. Generate Drugs
   
   python main.py --stage generate
   
   Output:
   • results/generation_results.json (with properties)
   • Console logging of generated molecules

FULL PIPELINE (ALL STAGES):

   python main.py --stage all --epochs 50 --batch-size 32

MONITORING TRAINING:

   Real-time logging shows:
   • Epoch number and progress
   • Training loss
   • Validation loss
   • Learning rate
   • Best model checkpoint indicator

   TensorBoard (if you add it):
   # tensorboard --logdir checkpoints/

HYPERPARAMETER TUNING:

   Edit CONFIG in main.py to adjust:
   
   # Model size
   'd_model': 512 → try 256 (small), 768 (large)
   'num_layers': 6 → try 4 (small), 8 (large)
   'num_heads': 8
   
   # Training
   'learning_rate': 3e-4 → try 1e-4, 1e-3
   'batch_size': 32 → try 16, 64
   'epochs': 50 → try 30, 100
   'warmup_steps': 4000
   
   # Data
   'train_split': 0.8 → use 0.7 for less training data
   
   # Protein encoding
   'protein_model': 'protbert' → try 'esm2' (better), 'prottrans'
"""

# ============================================================================
# 5. GENERATING DRUGS
# ============================================================================

"""
AFTER TRAINING, GENERATE DRUGS:

Basic Generation:
   python main.py --stage generate

The system will:
   1. Load trained model from checkpoints/best_model.pt
   2. Load protein embeddings from results/protein_embeddings.npz
   3. Select 10 random proteins
   4. Generate SMILES for each
   5. Validate with RDKit
   6. Calculate molecular properties
   7. Save to results/generation_results.json

GENERATION METHODS:

Change in main.py CONFIG:
   
   'generation_method': 'beam_search'  # Recommended for quality
   # OR 'greedy'  # Fast but lower quality
   # OR 'sample'  # Diverse outputs
   
   'beam_width': 5  # For beam search (1-10)

OUTPUT FORMAT:

results/generation_results.json:
{
  "protein_id": "A0AVT1",
  "generated_smiles": "CCO",
  "canonical_smiles": "CCO",
  "is_valid": true,
  "validation_message": "Valid SMILES",
  "score": 0.9523,
  "properties": {
    "molecular_weight": 46.04,
    "logp": -0.27,
    "num_h_donors": 1,
    "num_h_acceptors": 1,
    "num_rotatable_bonds": 0,
    "tpsa": 20.23
  }
}

INTERPRETING RESULTS:

molecular_weight: 300-500 is typical for drugs
logp: -1 to 5 is good (balance hydrophilic/hydrophobic)
h_donors: 0-5 (fewer is better)
h_acceptors: 0-10 (fewer is better)
rotatable_bonds: 0-10 (fewer is more rigid, better)
tpsa: 20-200 (lower = better oral bioavailability)
"""

# ============================================================================
# 6. ADVANCED USAGE
# ============================================================================

"""
PROGRAMMATIC USAGE:

from protein_encoder import ProteinEncoder, ProteinDatasetBuilder
from tokenizer import SMILESTokenizer
from inference import DrugGenerator, MolecularValidator
from data_loader import CPIDataLoader

# 1. Encode proteins
encoder = ProteinEncoder(model_name="protbert")
embedding = encoder.encode_sequence("MKFLKFSLLTAVLL...")

# 2. Tokenize SMILES
tokenizer = SMILESTokenizer()
tokenizer.build_vocab(["CCO", "CC(C)O", "CCCO"])
tokens = tokenizer.encode("CCO")

# 3. Generate drugs
generator = DrugGenerator(model, tokenizer)
smiles = generator.greedy_decode(protein_embedding)
smiles_list = generator.beam_search_decode(protein_embedding, beam_width=5)
samples = generator.sample_decode(protein_embedding, num_samples=10)

# 4. Validate
validator = MolecularValidator()
is_valid, msg = validator.validate_smiles("CCO")
canonical = validator.canonicalize_smiles("CCO")
props = validator.calculate_properties("CCO")

CUSTOM TRAINING LOOP:

from train import Trainer, TrainingConfig
from model import build_transformer

# Build model
transformer = build_transformer(...)
model = ProteinDrugTransformer(transformer, ...)

# Create trainer
trainer = Trainer(model, config, train_loader, val_loader)

# Train
trainer.train()

# Access results
history = trainer.get_history()

BATCH GENERATION:

# Generate for multiple proteins
embeddings = np.random.randn(100, 768)
smiles_list = generator.generate_batch(
    embeddings,
    method='beam_search',
    beam_width=5
)

CUSTOM MODEL CONFIGURATION:

In main.py, modify CONFIG:
{
    'd_model': 768,           # Larger model
    'num_layers': 8,          # More depth
    'num_heads': 12,          # More attention heads
    'd_ff': 3072,             # Larger FF layer
    'batch_size': 64,         # Larger batch
    'learning_rate': 2e-4,    # Slower learning
    'epochs': 100,            # Longer training
}
"""

# ============================================================================
# 7. TROUBLESHOOTING
# ============================================================================

"""
PROBLEM: Out of Memory (OOM)
SOLUTIONS:
    • Reduce batch_size: 32 → 16 or 8
    • Reduce d_model: 512 → 256
    • Reduce num_layers: 6 → 4
    • Reduce max_smiles_len: 512 → 256
    • Use CPU mode for debugging
    • Use smaller protein model

PROBLEM: Poor SMILES Quality
SOLUTIONS:
    • Train longer: 50 → 100 epochs
    • Use better protein encoder: protbert → esm2
    • Increase model size: d_model 512 → 768
    • Lower learning rate: 3e-4 → 1e-4
    • Check data quality (valid SMILES?)

PROBLEM: Slow Training
SOLUTIONS:
    • Use GPU: device = 'cuda'
    • Verify GPU is being used: 
      python -c "import torch; print(torch.cuda.is_available())"
    • Increase num_workers in dataloader
    • Reduce max_smiles_len
    • Reduce number of epochs for testing

PROBLEM: UniProt API Timeout
SOLUTIONS:
    • Check internet connection
    • Retry after some time
    • Sequences are cached, so 2nd run is faster
    • Manually prepare sequence file
    • Skip preprocessing if embeddings exist

PROBLEM: Import Errors
SOLUTIONS:
    • Verify installation: pip list
    • Reinstall requirements: pip install -r requirements.txt --force-reinstall
    • Check Python version: python --version
    • Activate virtual environment

PROBLEM: CUDA Out of Memory
SOLUTIONS:
    • Clear GPU memory: torch.cuda.empty_cache()
    • Reduce batch size significantly
    • Use CPU: device = 'cpu'
    • Check GPU with: nvidia-smi

COMMON ERRORS:

FileNotFoundError: '../CPI/CPI/smiles.smi'
    → Check data directory path
    → Verify files exist
    → Update data_dir in CONFIG

ModuleNotFoundError: 'esm'
    → Not critical, will use ProtBERT instead
    → Install with: pip install fair-esm

CUDA out of memory
    → Reduce batch_size or model size
    → Use CPU for initial testing

RuntimeError: size mismatch
    → Check embedding dimensions match d_model
    → Verify tokenizer vocabulary size
"""

# ============================================================================
# 8. PERFORMANCE OPTIMIZATION
# ============================================================================

"""
SPEED OPTIMIZATION:

1. Use Greedy Decoding (fastest)
   'generation_method': 'greedy'
   → ~100-200 SMILES/second per GPU

2. Reduce Model Size (training)
   'd_model': 256
   'num_layers': 4
   → ~2x faster training

3. Increase Batch Size (training)
   'batch_size': 64
   → Better GPU utilization
   → Faster epoch completion

4. Use Mixed Precision
   # If implemented:
   with autocast():
       logits = model(...)

5. Reduce Sequence Lengths
   'max_smiles_len': 256
   'max_protein_embedding_len': 512

QUALITY OPTIMIZATION:

1. Use Better Protein Encoder
   'protein_model': 'esm2'  # Slower but better
   
2. Increase Model Size
   'd_model': 768
   'num_layers': 8
   
3. Train Longer
   'epochs': 100  # Instead of 50
   
4. Use Beam Search
   'generation_method': 'beam_search'
   'beam_width': 10  # Higher = better but slower
   
5. Fine-tune Learning Rate
   'learning_rate': 1e-4  # Try different values

MEMORY OPTIMIZATION:

1. Gradient Accumulation (implement in train.py)
   → Simulate larger batch size with less memory
   
2. Checkpoint Activation
   → Save intermediate activations, recompute on backward
   
3. Use 16-bit Precision
   → Half memory, slight quality loss
   
4. Profile Memory Usage
   python -m torch.utils.bottleneck main.py

RECOMMENDED SETTINGS:

For Development/Testing:
    d_model: 256, num_layers: 4, batch_size: 16, epochs: 5

For Production/Research:
    d_model: 512, num_layers: 6, batch_size: 32, epochs: 100

For High-Performance:
    d_model: 768, num_layers: 8, batch_size: 64, epochs: 200
"""

# ============================================================================
# 9. EXPECTED RESULTS
# ============================================================================

"""
TYPICAL METRICS:

Training:
    Initial train loss: ~5-6 (vocab log scale)
    Final train loss: ~2-3
    Final val loss: ~2.5-3.5
    Training time: 2-4 hours on GPU
    
Generation:
    Valid SMILES: 70-85%
    Unique compounds: 90%+ (no duplicates)
    Average generation time: ~5-10ms per molecule
    
Chemical Quality:
    Molecular weight range: 200-600 Da
    LogP range: -2 to 6
    Drug-like (Lipinski's rule): 60-80%

EXAMPLE OUTPUT:

Generated Molecule for Protein P12345:
    Generated SMILES: CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O
    Canonical SMILES: CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O
    Valid: Yes
    Molecular Weight: 206.28 Da
    LogP: 3.97
    H-donors: 1
    H-acceptors: 2
    Rotatable Bonds: 3
    TPSA: 37.30 Ų
    
    → This looks like Ibuprofen! Good drug-like properties.

COMPARISON WITH KNOWN DRUGS:

Aspirin:
    MW: 180, LogP: 1.19, HBD: 1, HBA: 3
    
Ibuprofen:
    MW: 206, LogP: 3.97, HBD: 1, HBA: 2
    
Caffeine:
    MW: 194, LogP: 0.16, HBD: 0, HBA: 3
    
Generated examples should fall in similar ranges.
"""

# ============================================================================
# 10. REFERENCES
# ============================================================================

"""
PAPERS & RESOURCES:

Model Architecture:
    • Attention is All You Need (Vaswani et al., 2017)
    • https://arxiv.org/abs/1706.03762

Protein Encoders:
    • ProtBERT: https://github.com/agemagician/ProtTrans
    • ESM2 (Meta): https://github.com/facebookresearch/esm
    • ProtT5: https://github.com/agemagician/ProtTrans

Molecular Generation:
    • Recurrent Neural Network for Drug Discovery (Gómez-Bombarelli et al.)
    • Molecular Transformer (Schwaller et al., 2019)

Datasets:
    • ChEMBL: https://www.ebi.ac.uk/chembl/
    • UniProt: https://www.uniprot.org/
    • PDBbind: https://www.pdbbind.org.cn/

Tools:
    • RDKit: https://www.rdkit.org/
    • PyTorch: https://pytorch.org/
    • Hugging Face Transformers: https://huggingface.co/transformers/

TUTORIALS:

Getting Started:
    1. Run quickstart.py for examples
    2. Check README_PROTEIN_DRUG.md
    3. Review IMPLEMENTATION_SUMMARY.md
    4. Study ARCHITECTURE.md
    5. Read inline code comments

Deep Learning:
    • FastAI course
    • Stanford CS224N (NLP)
    • Stanford CS231N (Vision)

Chemistry:
    • SMILES tutorial
    • Molecular properties (Lipinski's rule)
    • Drug-likeness criteria

COMMUNITY:

    GitHub: Issues & discussions
    PyTorch Forums: Deep learning help
    ChemInformatics: Molecular modeling
    Stack Overflow: Programming help

CITATION:

If you use this code in research, please cite:
    
    @software{protein_drug_gen_2025,
        title={Protein-to-Drug Generation using Transformer},
        author={Your Name},
        year={2025},
        url={https://github.com/yourusername/repo}
    }
"""

# ============================================================================
# END OF GUIDE
# ============================================================================

if __name__ == "__main__":
    print(__doc__)
