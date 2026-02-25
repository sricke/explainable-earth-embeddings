import pandas as pd
import json
from pathlib import Path

if __name__ == "__main__":
    df = pd.read_csv('index_with_ee.csv')

    # extract all unique values from categorical columns
    categorical_columns = [
        # 'country',
        # 'state', 
        'biome',
        # 'ecoregion',
        'land_cover'
    ]

    vocab = []
    print("Extracting categorical values from dataset...")
    for col in categorical_columns:
        unique_values = df[col].dropna().unique()
        vocab.extend(unique_values.tolist())
        print(f"  {col}: {len(unique_values)} unique values")

    # add all categorized descriptors from the bins used in text generation
    categorized_descriptors = [
        # elevation categories
        "lowland",
        "platform/hill", 
        "mountain",
        
        # population density categories
        "rural area",
        "town or semi-dense area",
        "town or city",
        
        # temperature categories
        "polar",
        "cold temperate",
        "warm temperate",
        "tropical",
        
        # precipitation categories
        "hyper-arid",
        "arid",
        "semi-arid",
        "sub-humid",
        "humid",
        
        # tree cover categories
        "non-forest",
        "savanna/sparse vegetation",
        "woodland",
        "forest",
        
        # vegetation categories
        "bare soil or water",
        "very sparse vegetation",
        "sparse to moderate vegetation",
        "dense vegetation",
        "very dense vegetation"
    ]

    print(f"\nAdding {len(categorized_descriptors)} categorized descriptors...")

    vocab.extend(categorized_descriptors)

    vocab = sorted(list(set(vocab)))

    print(f"\n{'='*60}")
    print(f"Created complete vocabulary with {len(vocab)} unique concepts")
    print(f"{'='*60}")

    print(f"\nFirst 30 concepts:")
    for i, concept in enumerate(vocab[:30], 1):
        print(f"  {i:2d}. {concept}")

    print(f"\n... and {len(vocab) - 30} more concepts")

    output_path = Path('satellite_vocab_land_no_admin_ecoregion.json')
    with open(output_path, 'w') as f:
        json.dump(vocab, f, indent=2)

    print(f"\nSaved complete vocabulary to {output_path}")

    print(f"\nVocabulary Statistics:")
    print(f"  - Geographic entities: ~{len(vocab) - len(categorized_descriptors)}")
    print(f"  - Categorized descriptors: {len(categorized_descriptors)}")
    print(f"  - Total concepts: {len(vocab)}")