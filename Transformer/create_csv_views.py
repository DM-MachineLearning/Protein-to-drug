#!/usr/bin/env python
"""
Create simplified CSV views of training and generated data
"""

import pandas as pd
from pathlib import Path

print("\n" + "="*100)
print("CREATING SIMPLIFIED CSV VIEWS")
print("="*100 + "\n")

output_dir = Path("./results/training_run")

# ============================================================================
# 1. SIMPLIFIED TRAINING VIEW
# ============================================================================
print("Creating simplified training view...")

train_df = pd.read_csv(output_dir / "training_data.csv")

# Create a simpler view with just key columns
train_simple = train_df[['protein_id', 'smiles']].copy()
train_simple.columns = ['Protein_ID', 'Drug_SMILES']
train_simple.insert(0, 'Index', range(1, len(train_simple) + 1))

train_simple_file = output_dir / "training_simple.csv"
train_simple.to_csv(train_simple_file, index=False)

print(f"✓ Training data (simplified): {train_simple_file.name}")
print(f"  Rows: {len(train_simple)}")
print(f"  Columns: Index, Protein_ID, Drug_SMILES")
print()

# ============================================================================
# 2. SIMPLIFIED GENERATION VIEW
# ============================================================================
print("Creating simplified generation view...")

gen_df = pd.read_csv(output_dir / "generated_drugs.csv")

# Create a simpler view
gen_simple = gen_df[['protein_id', 'generated_smiles']].copy()
gen_simple.columns = ['Protein_ID', 'Generated_SMILES']
gen_simple.insert(0, 'Index', range(1, len(gen_simple) + 1))

gen_simple_file = output_dir / "generated_simple.csv"
gen_simple.to_csv(gen_simple_file, index=False)

print(f"✓ Generated data (simplified): {gen_simple_file.name}")
print(f"  Rows: {len(gen_simple)}")
print(f"  Columns: Index, Protein_ID, Generated_SMILES")
print()

# ============================================================================
# 3. COMBINED VIEW (All proteins with their data)
# ============================================================================
print("Creating combined view...")

# Training data with label
train_combined = train_simple.copy()
train_combined['Type'] = 'Training'
train_combined = train_combined[['Index', 'Protein_ID', 'Type', 'Drug_SMILES']]

# Generated data with label
gen_combined = gen_simple.copy()
gen_combined['Type'] = 'Generated'
gen_combined.columns = ['Index', 'Protein_ID', 'Type', 'Drug_SMILES']
gen_combined['Index'] = range(len(train_combined) + 1, len(train_combined) + len(gen_combined) + 1)

# Combine
combined = pd.concat([train_combined, gen_combined], ignore_index=True)

combined_file = output_dir / "all_proteins_combined.csv"
combined.to_csv(combined_file, index=False)

print(f"✓ Combined view: {combined_file.name}")
print(f"  Total Rows: {len(combined)}")
print(f"  Training rows: {len(train_combined)}")
print(f"  Generated rows: {len(gen_combined)}")
print(f"  Columns: Index, Protein_ID, Type, Drug_SMILES")
print()

# ============================================================================
# 4. STATISTICS VIEW
# ============================================================================
print("Creating statistics view...")

stats_data = {
    'Dataset': ['Training', 'Generated', 'Total'],
    'Protein_Count': [
        len(train_simple),
        len(gen_simple),
        len(train_simple) + len(gen_simple)
    ],
    'Unique_SMILES': [
        train_simple['Drug_SMILES'].nunique(),
        gen_simple['Generated_SMILES'].nunique(),
        combined['Drug_SMILES'].nunique()
    ],
    'Protein_ID_Range': [
        'P000000 - P000319',
        'P000320 - P000399',
        'P000000 - P000399'
    ]
}

stats_df = pd.DataFrame(stats_data)
stats_file = output_dir / "statistics.csv"
stats_df.to_csv(stats_file, index=False)

print(f"✓ Statistics view: {stats_file.name}")
print(stats_df.to_string(index=False))
print()

# ============================================================================
# 5. DETAILED SUMMARY
# ============================================================================
print("Creating detailed summary...")

summary_data = {
    'Metric': [
        'Total Proteins',
        'Training Proteins',
        'Test (Unseen) Proteins',
        'Training Drug-Protein Pairs',
        'Generated Molecules',
        'Unique Drugs in Training',
        'Unique Generated SMILES',
        'Total Protein Embeddings',
        'Embedding Dimension',
    ],
    'Value': [
        '400',
        '320',
        '80',
        '320',
        '80',
        '18',
        len(gen_simple['Generated_SMILES'].unique()),
        '400',
        '768',
    ]
}

summary_file = output_dir / "detailed_summary.csv"
summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(summary_file, index=False)

print(f"✓ Detailed summary: {summary_file.name}")
print(summary_df.to_string(index=False))
print()

# ============================================================================
# 6. PROTEIN MAPPING VIEW
# ============================================================================
print("Creating protein mapping view...")

# Create a view showing each protein and its associated data
mapping_data = []

# Add training proteins
for idx, row in train_simple.iterrows():
    mapping_data.append({
        'Protein_ID': row['Protein_ID'],
        'Index': row['Index'],
        'Dataset': 'Training',
        'SMILES': row['Drug_SMILES'],
        'Status': 'Known'
    })

# Add test proteins
for idx, row in gen_simple.iterrows():
    mapping_data.append({
        'Protein_ID': row['Protein_ID'],
        'Index': row['Index'],
        'Dataset': 'Generated',
        'SMILES': row['Generated_SMILES'],
        'Status': 'Generated'
    })

mapping_df = pd.DataFrame(mapping_data)
mapping_file = output_dir / "protein_mapping.csv"
mapping_df.to_csv(mapping_file, index=False)

print(f"✓ Protein mapping: {mapping_file.name}")
print(f"  Rows: {len(mapping_df)}")
print(f"  Columns: Protein_ID, Index, Dataset, SMILES, Status")
print()

# ============================================================================
# 7. QUICK LOOKUP TABLE
# ============================================================================
print("Creating quick lookup table...")

lookup_data = []

# Training data
for _, row in train_simple.head(5).iterrows():
    lookup_data.append({
        'Protein': row['Protein_ID'],
        'Type': 'TRAIN',
        'Molecule': row['Drug_SMILES']
    })

# Generated data
for _, row in gen_simple.head(5).iterrows():
    lookup_data.append({
        'Protein': row['Protein_ID'],
        'Type': 'GEN',
        'Molecule': row['Generated_SMILES']
    })

lookup_df = pd.DataFrame(lookup_data)
lookup_file = output_dir / "quick_lookup.csv"
lookup_df.to_csv(lookup_file, index=False)

print(f"✓ Quick lookup (sample): {lookup_file.name}")
print(lookup_df.to_string(index=False))
print()

# ============================================================================
# DISPLAY ALL FILES
# ============================================================================
print("\n" + "="*100)
print("✅ ALL SIMPLIFIED CSV FILES CREATED")
print("="*100 + "\n")

print("📊 NEW CSV FILES:")
print(f"  1. training_simple.csv")
print(f"     - Simplified training data (Protein_ID + Drug_SMILES)")
print(f"     - {len(train_simple)} rows")
print()
print(f"  2. generated_simple.csv")
print(f"     - Simplified generated data (Protein_ID + Generated_SMILES)")
print(f"     - {len(gen_simple)} rows")
print()
print(f"  3. all_proteins_combined.csv")
print(f"     - All 400 proteins in one view with type labels")
print(f"     - {len(combined)} rows")
print()
print(f"  4. statistics.csv")
print(f"     - Quick statistics comparing training vs generated")
print()
print(f"  5. detailed_summary.csv")
print(f"     - Detailed metrics and counts")
print()
print(f"  6. protein_mapping.csv")
print(f"     - Full mapping of all proteins with their molecules")
print(f"     - {len(mapping_df)} rows")
print()
print(f"  7. quick_lookup.csv")
print(f"     - Sample data for quick reference (first 5 of each)")
print()

print("📁 LOCATION: results/training_run/")
print()
print("✨ All files are ready for easy viewing and analysis!")
