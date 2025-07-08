# math_agent_query.py
# Query the math knowledge base in Qdrant for similar problems and solutions

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

# 1. Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Connect to Qdrant
client = QdrantClient("localhost", port=6333)
collection_name = "math_problems"

# 3. Define a function to query the knowledge base
def query_math_agent(user_query, top_k=3):
    query_embedding = model.encode([user_query])[0]
    results = client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=top_k
    )
    return [
        {
            "question": r.payload["question"],
            "answer": r.payload["answer"],
            "topic": r.payload.get("topic", "unknown"),
            "difficulty": r.payload.get("difficulty", "unknown"),
            "score": r.score
        }
        for r in results
    ]

if __name__ == "__main__":
    query = input("Enter your math question: ")
    results = query_math_agent(query)
    for idx, res in enumerate(results, 1):
        print(f"\nResult {idx}:")
        print("Question:", res["question"])
        print("Answer:", res["answer"])
        print("Topic:", res["topic"])
        print("Difficulty:", res["difficulty"])
        print("Score:", res["score"])
