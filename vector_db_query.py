# Minimal example: Query Qdrant for similar GSM8K questions

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

# 1. Define your query
query = "How many apples are left after giving away some?"

# 2. Create embedding for the query
model = SentenceTransformer('all-MiniLM-L6-v2')
query_embedding = model.encode([query])[0]

# 3. Connect to Qdrant
client = QdrantClient("localhost", port=6333)
collection_name = "gsm8k_questions"

# 4. Search for the most similar question
search_result = client.search(
    collection_name=collection_name,
    query_vector=query_embedding,
    limit=1
)

# 5. Print the most similar question
if search_result:
    print("Most similar question:", search_result[0].payload["question"])
    print("Score:", search_result[0].score)
else:
    print("No results found.")
