import os
import subprocess
import json

def generate_concepts_with_ollama(category, num_concepts=50):
    """
    Use Ollama to systematically generate concepts for a category
    """
    prompt = f"""You must return ONLY a valid Python list. No explanation, no introduction, no numbering.

    Generate {num_concepts} specific, visually distinctive concepts related to {category} 
    that would be recognizable in satellite imagery.

    Requirements:
    - Each concept should be 1-2 words
    - Focus on things visible from above
    - Be general and specific (e.g., "crop field" and "field")
    - Cover range of scales, densities, and subtypes
    - No duplicates

    Your response must contain {num_concepts} concepts.
    Your response must start with [ and end with ]
    Example format: ["concept 1", "concept 2", "concept 3"]

    Python list:"""

    ollama_path = os.path.expanduser("~/ollama/bin/ollama")

    result = subprocess.run(
        [ollama_path, 'run', 'llama3.1'],
        input=prompt,
        text=True,
        capture_output=True
    )

    response_text = result.stdout.strip()

    try:
        # Convert the response to a Python list
        concepts = eval(response_text)
        if not isinstance(concepts, list):
            raise ValueError("Response is not a list")
    except Exception as e:
        print(f"Failed to parse Ollama response for {category}:", e)
        print(f"Response was: {response_text[:200]}")
        concepts = []

    return concepts

def normalize_concept(text):
    """Normalize a concept to consistent format (lowercase)"""
    text = ' '.join(text.split()).strip()
    text = text.lower()
    return text


categories = [
    "urban infrastructure",
    "agricultural patterns",
    "natural vegetation",
    "water bodies and features",
    "transportation networks",
    "industrial facilities",
    "geological features",
    "coastal and marine features"
]

all_generated = []
for category in categories:
    print(f"Generating concepts for: {category}")
    concepts = generate_concepts_with_ollama(category, num_concepts=75)
    
    normalized = [normalize_concept(concept) for concept in concepts]
    all_generated.extend(normalized)
    
    print(f"  → Generated {len(concepts)} concepts")

print(f"\nTotal generated: {len(all_generated)}")

# remove duplicates
seen = set()
unique_concepts = []
for concept in all_generated:
    if concept not in seen and concept:  # also filter empty strings
        seen.add(concept)
        unique_concepts.append(concept)

print(f"Unique concepts: {len(unique_concepts)}")

output_data = {
    "categories": categories,
    "total_concepts": len(unique_concepts),
    "concepts": sorted(unique_concepts)
}

output_file = "generated_concepts.json"
with open(output_file, 'w') as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)

print(f"\n✓ Saved {len(unique_concepts)} unique concepts to {output_file}")

text_output_file = "generated_concepts.txt"
with open(text_output_file, 'w') as f:
    for concept in sorted(unique_concepts):
        f.write(f"{concept}\n")

print(f"✓ Saved to {text_output_file}")