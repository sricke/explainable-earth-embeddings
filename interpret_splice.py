import pandas as pd

def interpret_location_index(df, idx):
    location = df.iloc[idx]
    concept_weights = location.drop(['lat', 'lon', 'fn', 'cosine_similarity'])
    concept_weights = pd.to_numeric(concept_weights, errors="coerce")

    top_concepts = concept_weights.nlargest(10)
    print(f"Location: {location['lat']}, {location['lon']}")
    print(f"Reconstruction quality: {location['cosine_similarity']:.3f}")
    print("\nTop 10 concepts:")
    print(top_concepts)

def check_reconstruction_quality(df):
    print(f"Mean cosine similarity: {df['cosine_similarity'].mean():.3f}")
    print(f"Min: {df['cosine_similarity'].min():.3f}")
    print(f"Max: {df['cosine_similarity'].max():.3f}")

def find_most_common_concepts(df):
    concept_cols = df.columns.drop(['lat', 'lon', 'fn', 'cosine_similarity'])
    concept_sparsity = (df[concept_cols] > 0).sum()
    most_common = concept_sparsity.nlargest(20)
    print("Most frequently used concepts:")
    print(most_common)

def analyze_sparsity(df):
    avg_nonzero = (df[concept_cols] > 0).sum(axis=1).mean()
    print(f"Average number of active concepts per location: {avg_nonzero:.1f}")

if __name__ == "__main__":

    df = pd.read_parquet('location_weights_batch_1.parquet')