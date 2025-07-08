# Minimal example: Store GSM8K questions in Qdrant with embeddings

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance


# 1. Load the full GSM8K main train split
print("Loading full GSM8K 'main' train split...")
gsm8k = load_dataset('gsm8k', 'main', split='train')
gsm8k_questions = [item['question'] for item in gsm8k]


# 1b. Load the Math-QA dataset (https://huggingface.co/datasets/rvv-karma/Math-QA)
print("Loading Math-QA dataset (rvv-karma/Math-QA)...")
math_qa = load_dataset('rvv-karma/Math-QA', split='train')
math_qa_questions = [item['question'] for item in math_qa]

# Combine all questions
questions = gsm8k_questions + math_qa_questions



# 2. Create embeddings (use GPU if available)
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
embeddings = model.encode(questions, batch_size=64, show_progress_bar=True)

# 3. Connect to Qdrant (local, default port)
client = QdrantClient("localhost", port=6333)

# 4. Create a collection (if not exists)
collection_name = "gsm8k_questions"
client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=embeddings.shape[1], distance=Distance.COSINE)
)


# 5. Upload all data in batches
payload = []
for item in gsm8k:
    payload.append({"question": item['question'], "source": "gsm8k"})
for item in math_qa:
    payload.append({"question": item['question'], "source": "math_qa"})
client.upload_collection(
    collection_name=collection_name,
    vectors=embeddings,
    payload=payload,
    ids=None,  # auto-generate ids
    batch_size=64
)


# Check how many points are in the collection after upload
count = client.count(collection_name=collection_name).count
print(f"Uploaded {len(questions)} questions to Qdrant!")
print(f"Qdrant collection '{collection_name}' now contains {count} vectors.")
