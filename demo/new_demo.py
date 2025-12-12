import warnings
import torch
import pandas as pd
import numpy as np
import faiss
import joblib
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

warnings.filterwarnings('ignore')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

clf = joblib.load('../models/logistic_regression_model.pkl')
vectorizer = joblib.load('../models/tfidf_vectorizer.pkl')

train = pd.read_csv('../data/train_lemmatized.csv').dropna(subset=['message_lemmatized', 'label'])
documents = train['message_lemmatized'].tolist()
doc_labels = train['label'].tolist()

embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

bert_classifier = pipeline(
    "text-classification",
    model="MariaOnyshchuk/prop_xlmroberta",
    device=0 if device == 'cuda' else -1,
    return_all_scores=True
)

print("Loading cached FAISS index and embeddings...")
embeddings = np.load("train_embeddings.npy")
index = faiss.read_index("faiss_index.idx")

def batch_retrieve(texts, top_k=3):
    query_embeddings = embedding_model.encode(texts, batch_size=32, show_progress_bar=False)
    distances, indices = index.search(query_embeddings.astype('float32'), top_k)

    batch_retrieved = []
    for i in range(len(texts)):
        retrieved = []
        for idx, dist in zip(indices[i], distances[i]):
            retrieved.append({
                'text': documents[idx],
                'label': doc_labels[idx],
                'distance': float(dist)
            })
        batch_retrieved.append(retrieved)
    return batch_retrieved

def ensemble_classify(texts, batch_size=8):
    retrieved_docs_list = batch_retrieve(texts, top_k=3)
    bert_probs_all = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_retrieved = retrieved_docs_list[i:i+batch_size]

        augmented_texts = []
        for text, retrieved in zip(batch_texts, batch_retrieved):
            context = "\n".join([doc['text'][:100] for doc in retrieved[:2]])
            augmented_texts.append(f"{context}\n\n{text}")

        batch_results = bert_classifier(augmented_texts, truncation=True, max_length=512)

        for result in batch_results:
            probs = np.array([score['score'] for score in result])
            bert_probs_all.append(probs)

    bert_probs_all = np.array(bert_probs_all)

    X_test = vectorizer.transform(texts)
    lr_probs = clf.predict_proba(X_test)

    ensemble_probs = 0.1 * bert_probs_all + 0.9 * lr_probs

    label_names = {0: 'misinformation', 1: 'propaganda', 2: 'opposition'}
    results = []

    for i in range(len(texts)):
        final_idx = np.argmax(ensemble_probs[i])
        results.append({
            'final_label': label_names[final_idx],
            'confidence': float(ensemble_probs[i][final_idx]),
            'bert_probs': bert_probs_all[i].tolist(),
            'lr_probs': lr_probs[i].tolist(),
            'retrieved': retrieved_docs_list[i]
        })

    return results

print("Loading cached RAG embeddings...")
df_events = pd.read_csv("../data/all_events.csv").dropna(subset=["description"])
rag_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', device=device)
event_embeddings = np.load("event_embeddings.npy")

def find_closest_event(text):
    query_emb = rag_model.encode([text], convert_to_tensor=True)
    sims = util.cos_sim(query_emb, event_embeddings)[0]

    top_idx = sims.argmax().item()
    return {
        "closest_text": df_events.iloc[top_idx]["description"],
        "similarity": sims[top_idx].item()
    }

print("\n*** Russian Propaganda Detection Terminal ***")
print("Type text to classify. Type 'exit' to quit.\n")

while True:
    user_input = input("\nEnter text: ").strip()

    if user_input.lower() == "exit":
        print("Exiting.")
        break
    if len(user_input) < 5:
        print("Please enter a longer text.")
        continue

    # run classifier
    result = ensemble_classify([user_input])[0]

    # run rag search
    rag_result = find_closest_event(user_input)

    print("\n--- Classification Result ---")
    print(f"Label: {result['final_label']}")
    print(f"Confidence: {result['confidence']:.4f}")

    print("\n--- FAISS Nearest Neighbors (training set) ---")
    for i, doc in enumerate(result['retrieved']):
        print(f"{i+1}. [{doc['label']}] dist={doc['distance']:.4f}")
        print(f"   {doc['text'][:150]}...")

    print("\n--- RAG: Closest Event From all_events.csv ---")
    print(f"Similarity: {rag_result['similarity']:.4f}")
    print(f"Event: {rag_result['closest_text'][:250]}...")

    print("\n---------------------------------------------")
