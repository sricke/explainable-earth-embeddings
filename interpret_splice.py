import pandas as pd

def interpret_location_index(df, idx):
    location = df.iloc[idx]
    concept_weights = location.drop(['lat', 'lon', 'fn', 'mse'])
    concept_weights = pd.to_numeric(concept_weights, errors="coerce")

    top_concepts = concept_weights.nlargest(10)
    print(f"Location: {location['lat']}, {location['lon']}")
    print(f"Reconstruction quality: {location['mse']:.3f}")
    print("\nTop 10 concepts:")
    print(top_concepts)

def check_reconstruction_quality(df):
    print(f"Mean cosine similarity: {df['mse'].mean():.3f}")
    print(f"Min: {df['mse'].min():.3f}")
    print(f"Max: {df['mse'].max():.3f}")

def find_most_common_concepts(df):
    concept_cols = df.columns.drop(['lat', 'lon', 'fn', 'mse'])
    concept_sparsity = (df[concept_cols] > 0).sum()
    most_common = concept_sparsity.nlargest(20)
    print("Most frequently used concepts:")
    print(most_common)

def analyze_sparsity(df):
    concept_cols = df.columns.drop(['lat', 'lon', 'fn', 'mse'])
    avg_nonzero = (df[concept_cols] > 0).sum(axis=1).mean()
    print(f"Average number of active concepts per location: {avg_nonzero:.1f}")

def analyze_specific_concept(df, concept_name, top_n=None, threshold=0.5):
    if concept_name not in df.columns:
        available_concepts = [col for col in df.columns 
                             if col not in ['lat', 'lon', 'fn', 'mse']]
        raise ValueError(f"Concept '{concept_name}' not found. "
                        f"Available concepts: {len(available_concepts)}")
    
    activated = df[df[concept_name] > threshold].copy()
    
    activated = activated.sort_values(concept_name, ascending=False)
    
    if top_n is not None:
        activated = activated.head(top_n)
    
    print(f"\n{'='*60}")
    print(f"Concept Activation Analysis: '{concept_name}'")
    print(f"{'='*60}")
    print(f"Total locations in dataset: {len(df)}")
    print(f"Locations with {concept_name} activated (>{threshold}): {len(activated)}")
    print(f"Activation rate: {len(activated)/len(df)*100:.2f}%")
    print(f"\nWeight statistics:")
    print(f"  Mean weight: {activated[concept_name].mean():.4f}")
    print(f"  Median weight: {activated[concept_name].median():.4f}")
    print(f"  Max weight: {activated[concept_name].max():.4f}")
    print(f"  Min weight: {activated[concept_name].min():.4f}")
    print(f"  Std dev: {activated[concept_name].std():.4f}")
    
    return activated

if __name__ == "__main__":

    df = pd.read_parquet('location_weights_batch_1_l1_penalty_0.05_num_samples_10000.parquet')
    from IPython import embed; embed()