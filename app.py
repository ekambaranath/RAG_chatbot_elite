# app.py (lightweight offline RAG)
import pickle
from sentence_transformers import SentenceTransformer
import faiss

# Load FAISS index and paragraphs
with open("faiss_paragraphs.pkl", "rb") as f:
    paragraphs = pickle.load(f)
index = faiss.read_index("faiss_index.idx")

# Embedding model
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Simple appointment storage
appointments = []

def retrieve(query, top_k=3):
    q_emb = embed_model.encode([query])
    distances, indices = index.search(q_emb, top_k)
    results = [paragraphs[i] for i in indices[0]]
    return results

def elitebody_chat(query):
    if "book" in query.lower() or "appointment" in query.lower():
        name = input("Enter your name: ")
        treatment = input("Preferred treatment: ")
        date = input("Preferred date: ")
        time = input("Preferred time: ")
        appointments.append({
            "name": name, "treatment": treatment, "date": date, "time": time
        })
        return f"✅ Appointment booked for {name} for {treatment} on {date} at {time}."
    
    # Retrieve top relevant paragraphs
    results = retrieve(query, top_k=3)
    return "\n\n".join(results) if results else "Sorry, no info found."

# Demo
if __name__ == "__main__":
    print("Elite Body Home Chatbot (Lightweight Offline Version)")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            break
        response = elitebody_chat(user_input)
        print("\nBot:", response)

