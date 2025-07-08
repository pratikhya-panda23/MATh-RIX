# Math Knowledge Base Ingestion: GSM8K + MathQA

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
import numpy as np

# 1. Load GSM8K (full train split)
gsm8k = load_dataset('gsm8k', 'main', split='train')
gsm8k_questions = [item['question'] for item in gsm8k]
gsm8k_answers = [item['answer'] for item in gsm8k]
gsm8k_topics = ['grade_school'] * len(gsm8k_questions)
gsm8k_difficulties = ['easy'] * len(gsm8k_questions)

# 2. Load MathQA (filtered for valid questions/answers)
mathqa = load_dataset('math_qa', split='train', trust_remote_code=True)
# Print keys to inspect structure
print('MathQA sample keys:', mathqa[0].keys())
# Use correct keys for MathQA
mathqa_questions = [item['Problem'] for item in mathqa if item.get('Problem') and item.get('Rationale')]
mathqa_answers = [item['Rationale'] for item in mathqa if item.get('Problem') and item.get('Rationale')]
mathqa_topics = [item['category'] if item.get('category') else 'unknown' for item in mathqa if item.get('Problem') and item.get('Rationale')]
mathqa_difficulties = ['medium'] * len(mathqa_questions)

# 3. Combine datasets
all_questions = gsm8k_questions + mathqa_questions
all_answers = gsm8k_answers + mathqa_answers
all_topics = gsm8k_topics + mathqa_topics
all_difficulties = gsm8k_difficulties + mathqa_difficulties

# 4. Create embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(all_questions, show_progress_bar=True)

# 5. Connect to Qdrant
client = QdrantClient("localhost", port=6333)
collection_name = "math_problems"

# 6. (Re)create collection
client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=embeddings.shape[1], distance=Distance.COSINE)
)

# 7. Upload data in batches
batch_size = 100
for i in range(0, len(all_questions), batch_size):
    batch_vectors = embeddings[i:i+batch_size]
    batch_payload = [
        {
            "question": all_questions[j],
            "answer": all_answers[j],
            "topic": all_topics[j],
            "difficulty": all_difficulties[j]
        }
        for j in range(i, min(i+batch_size, len(all_questions)))
    ]
    client.upload_collection(
        collection_name=collection_name,
        vectors=batch_vectors,
        payload=batch_payload,
        ids=None,
        batch_size=batch_size
    )

print(f"Uploaded {len(all_questions)} math problems (GSM8K + MathQA) to Qdrant!")
