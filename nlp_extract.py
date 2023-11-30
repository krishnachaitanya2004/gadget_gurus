import nltk
import numpy as np
import pandas as pd
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')

# Load the WordNet Lemmatizer
lemmatizer = WordNetLemmatizer()

feature_keywords = {
    "price": ["price", "cost", "budget-friendly"],
    "sim": ["sim", "card", "network", "connectivity"],
    "processor": ["processor", "CPU", "performance", "speed", "gaming", "multitasking"],
    "ram": ["ram", "memory", "storage", "speed", "gaming", "multitasking"],
    "storage": ["storage", "capacity", "space", "memory", "expandable"],
    "battery": ["battery", "power", "life", "charging", "fast-charging"],
    "os": ["os", "operating system", "software"],
    "cam": ["camera", "cam", "photography", "photographic", "photos", "video", "resolution", "low-light", "selfie"],
    "display": ["display", "screen", "resolution", "size", "movie"]
}

feature_adjectives = {
    "price": {"expensive": 5, "affordable": 2, "budget-friendly": 3},
    "processor": ["fast", "powerful", "efficient"],
    "ram": ["high", "adequate", "sufficient"],
    "storage": ["spacious", "sufficient", "expandable"],
    "battery": ["long-lasting", "efficient", "powerful"],
    "os": ["user-friendly", "latest", "efficient"],
    "cam": ["spectacular", "clear", "blur"],
    "display": ["vibrant", "sharp", "large", "crisp"]
}

feature_order = ['price', 'sim', 'processor', 'cam', 'ram', 'storage', 'battery', 'os', 'display']

# Define words for comparison
comparison_words = ["best", "good", "average", "okay"]

def compute_similarity(word1, word2):
    """
    Placeholder for similarity computation in NLTK.
    """
    # This is just a placeholder; you might need a different approach for similarity in NLTK
    return 0.5

def assign_value_based_on_similarity(adjective, feature_adj):
    similarities = [compute_similarity(adjective, comp_word) for comp_word in comparison_words]
    maxi_ind = np.argmax(np.array(similarities))
    return 5 - maxi_ind

def extract_features(query):
    # Tokenize and lemmatize the query using NLTK
    tokens = word_tokenize(query)
    lemmatized_tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens]

    # Initialize feature vectors
    feature_vectors = {}

    # Extract adjectives and assign feature values based on the adjectives
    for i in range(len(lemmatized_tokens)):
        token = lemmatized_tokens[i]
        if token in feature_keywords.values():
            noun = token

            for feature, keywords in feature_keywords.items():
                if noun in keywords:
                    # Check adjectives attached to the noun or its neighboring words
                    adjectives = [lemmatized_tokens[j] for j in range(i - 1, i + 2) if lemmatized_tokens[j] in feature_adjectives.get(feature, [])]
                    for adj in adjectives:
                        feature_vectors[feature] = assign_value_based_on_similarity(adj, feature_adjectives.get(feature, []))

    # Assign default feature values (3) for features not mentioned in the query
    for feature in feature_keywords.keys():
        if feature not in feature_vectors:
            feature_vectors[feature] = 3

    ordered_list = [feature_vectors.get(feature, None) for feature in feature_order]
    return ordered_list

# Example usage:
queries = [
    "Looking for good cam phones",
    "Recommend phones with good battery life",
    # ... (other queries)
]

# Extract features for each query
feature_vectors = [extract_features(query) for query in queries]

# Create a DataFrame from the feature vectors
df = pd.DataFrame(feature_vectors, columns=feature_order)

# Display the DataFrame
print(df)
