import pickle
from sentence_transformers import SentenceTransformer
import faiss

# Load text from data folder
with open("data/elitebody.txt", "r") as f:
    text = f.read()

# Split text into paragraphs
paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

# Create embeddings locally
model = SentenceTransformer('all-MiniLM-L6-v2')  # small, fast, local
embeddings = model.encode(paragraphs)

# Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save index and paragraphs
with open("faiss_paragraphs.pkl", "wb") as f:
    pickle.dump(paragraphs, f)
faiss.write_index(index, "faiss_index.idx")

print(f"Ingested {len(paragraphs)} paragraphs and created FAISS index")

