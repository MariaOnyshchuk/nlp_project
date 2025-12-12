from flask import Flask, render_template_string, request, jsonify
import warnings
import torch
import pandas as pd
import numpy as np
import faiss
import joblib
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

warnings.filterwarnings('ignore')

app = Flask(__name__)

# Initialize models and data
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load models
clf = joblib.load('../models/logistic_regression_model.pkl')
vectorizer = joblib.load('../models/tfidf_vectorizer.pkl')

# Load training data
train = pd.read_csv('../data/train_lemmatized.csv').dropna(subset=['message_lemmatized', 'label'])
documents = train['message_lemmatized'].tolist()
doc_labels = train['label'].tolist()

# Load embedding models
embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

bert_classifier = pipeline(
    "text-classification",
    model="MariaOnyshchuk/prop_xlmroberta",
    device=0 if device == 'cuda' else -1,
    return_all_scores=True
)

# Load FAISS index
print("Loading cached FAISS index and embeddings...")
embeddings = np.load("train_embeddings.npy")
faiss_index = faiss.read_index("faiss_index.idx")

# Load RAG data
print("Loading cached RAG embeddings...")
df_events = pd.read_csv("../data/all_events.csv").dropna(subset=["description"])
rag_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', device=device)
event_embeddings = np.load("event_embeddings.npy")

def batch_retrieve(texts, top_k=3):
    query_embeddings = embedding_model.encode(texts, batch_size=32, show_progress_bar=False)
    distances, indices = faiss_index.search(query_embeddings.astype('float32'), top_k)

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

def find_closest_event(text):
    query_emb = rag_model.encode([text], convert_to_tensor=True)
    sims = util.cos_sim(query_emb, event_embeddings)[0]

    top_idx = sims.argmax().item()
    return {
        "closest_text": df_events.iloc[top_idx]["description"],
        "similarity": float(sims[top_idx].item())
    }

# HTML Template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Propaganda Detection System</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1e293b 0%, #1e40af 50%, #1e293b 100%);
            min-height: 100vh;
            padding: 2rem;
            color: #fff;
        }
        .container { max-width: 1000px; margin: 0 auto; }
        .header {
            text-align: center;
            margin-bottom: 2rem;
        }
        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.75rem;
        }
        .header p { color: #bfdbfe; font-size: 0.95rem; }
        .card {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 1rem;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        label {
            display: block;
            font-weight: 600;
            margin-bottom: 0.75rem;
            font-size: 1rem;
        }
        textarea {
            width: 100%;
            padding: 1rem;
            border-radius: 0.75rem;
            border: 2px solid #93c5fd;
            font-size: 1rem;
            resize: vertical;
            min-height: 120px;
            background: rgba(255, 255, 255, 0.95);
        }
        textarea:focus { outline: none; border-color: #3b82f6; }
        button {
            background: #2563eb;
            color: white;
            border: none;
            padding: 0.875rem 2rem;
            font-size: 1rem;
            font-weight: 600;
            border-radius: 0.75rem;
            cursor: pointer;
            margin-top: 1rem;
            width: 100%;
            transition: all 0.2s;
        }
        button:hover { background: #1d4ed8; transform: translateY(-1px); }
        button:disabled {
            background: #6b7280;
            cursor: not-allowed;
            transform: none;
        }
        #results { display: none; }
        .result-card {
            background: white;
            color: #1f2937;
            border-radius: 1rem;
            padding: 1.5rem;
            margin-bottom: 1rem;
        }
        .result-card h2 {
            font-size: 1.25rem;
            margin-bottom: 1rem;
            color: #1f2937;
        }
        .label-badge {
            display: inline-block;
            padding: 0.5rem 1.25rem;
            border-radius: 2rem;
            font-weight: 700;
            font-size: 1.1rem;
            border: 2px solid;
            margin-bottom: 1rem;
        }
        .propaganda { background: #fee2e2; color: #991b1b; border-color: #fca5a5; }
        .misinformation { background: #fed7aa; color: #9a3412; border-color: #fdba74; }
        .opposition { background: #dbeafe; color: #1e40af; border-color: #93c5fd; }
        .confidence-bar {
            width: 100%;
            height: 1rem;
            background: #e5e7eb;
            border-radius: 0.5rem;
            overflow: hidden;
            margin: 0.5rem 0;
        }
        .confidence-fill {
            height: 100%;
            background: #2563eb;
            transition: width 0.5s;
        }
        .probs-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
            margin-top: 1rem;
            padding: 1rem;
            background: #f9fafb;
            border-radius: 0.5rem;
        }
        .prob-section h4 {
            font-size: 0.75rem;
            color: #6b7280;
            margin-bottom: 0.5rem;
            text-transform: uppercase;
        }
        .prob-section div { font-size: 0.875rem; margin-bottom: 0.25rem; }
        .neighbor {
            padding: 0.875rem;
            background: #f9fafb;
            border: 1px solid #e5e7eb;
            border-radius: 0.5rem;
            margin-bottom: 0.75rem;
        }
        .neighbor-header {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 0.5rem;
        }
        .neighbor-badge {
            padding: 0.25rem 0.75rem;
            border-radius: 1rem;
            font-size: 0.75rem;
            font-weight: 600;
        }
        .neighbor-text {
            font-size: 0.875rem;
            color: #4b5563;
            margin-top: 0.5rem;
        }
        .rag-box {
            padding: 1rem;
            background: #eff6ff;
            border: 1px solid: #bfdbfe;
            border-radius: 0.5rem;
        }
        .rag-similarity {
            font-weight: 600;
            color: #1e40af;
            margin-bottom: 0.5rem;
        }
        .rag-text { font-size: 0.875rem; color: #374151; }
        .loader {
            display: inline-block;
            width: 1rem;
            height: 1rem;
            border: 2px solid rgba(255,255,255,0.3);
            border-top-color: white;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
            margin-right: 0.5rem;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        .footer {
            text-align: center;
            margin-top: 2rem;
            color: #bfdbfe;
            font-size: 0.875rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🛡️ Propaganda Detection System</h1>
            <p>AI-powered ensemble classifier with RAG for detecting Russian propaganda and misinformation</p>
        </div>

        <div class="card">
            <label for="textInput">Enter text to analyze:</label>
            <textarea id="textInput" placeholder="Type or paste text here for classification..."></textarea>
            <button onclick="analyzeText()" id="analyzeBtn">Analyze Text</button>
        </div>

        <div id="results"></div>

        <div class="footer">
            <p>Ensemble model: XLM-RoBERTa + Logistic Regression with FAISS retrieval</p>
        </div>
    </div>

    <script>
        async function analyzeText() {
            const text = document.getElementById('textInput').value.trim();
            
            if (text.length < 5) {
                alert('Please enter at least 5 characters');
                return;
            }

            const btn = document.getElementById('analyzeBtn');
            btn.disabled = true;
            btn.innerHTML = '<span class="loader"></span>Analyzing...';

            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text })
                });

                const data = await response.json();
                displayResults(data);
            } catch (error) {
                alert('Error analyzing text: ' + error.message);
            } finally {
                btn.disabled = false;
                btn.innerHTML = 'Analyze Text';
            }
        }

        function displayResults(data) {
            const resultsDiv = document.getElementById('results');
            
            if (!data || !data.final_label) {
                alert('Invalid response from server');
                return;
            }
            
            const labelClass = data.final_label;
            
            resultsDiv.innerHTML = `
                <div class="result-card">
                    <h2>📋 Classification Result</h2>
                    <span class="label-badge ${labelClass}">${data.final_label.toUpperCase()}</span>
                    <div>
                        <strong>Confidence:</strong>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: ${data.confidence * 100}%"></div>
                        </div>
                        <strong>${(data.confidence * 100).toFixed(1)}%</strong>
                    </div>
                    
                    <div class="probs-grid">
                        <div class="prob-section">
                            <h4>BERT Probabilities</h4>
                            <div>Misinformation: ${(data.bert_probs[0] * 100).toFixed(1)}%</div>
                            <div>Propaganda: ${(data.bert_probs[1] * 100).toFixed(1)}%</div>
                            <div>Opposition: ${(data.bert_probs[2] * 100).toFixed(1)}%</div>
                        </div>
                        <div class="prob-section">
                            <h4>Logistic Regression</h4>
                            <div>Misinformation: ${(data.lr_probs[0] * 100).toFixed(1)}%</div>
                            <div>Propaganda: ${(data.lr_probs[1] * 100).toFixed(1)}%</div>
                            <div>Opposition: ${(data.lr_probs[2] * 100).toFixed(1)}%</div>
                        </div>
                    </div>
                </div>

                <div class="result-card">
                    <h2>📊 FAISS Nearest Neighbors</h2>
                    ${data.retrieved.map((doc, i) => `
                        <div class="neighbor">
                            <div class="neighbor-header">
                                <span class="neighbor-badge ${doc.label}">${doc.label}</span>
                                <span style="font-size: 0.875rem; color: #6b7280;">
                                    Distance: ${doc.distance.toFixed(4)}
                                </span>
                            </div>
                            <div class="neighbor-text">${doc.text.substring(0, 150)}...</div>
                        </div>
                    `).join('')}
                </div>

                <div class="result-card">
                    <h2>🔍 RAG: Closest Historical Event</h2>
                    <div class="rag-box">
                        <div class="rag-similarity">Similarity: ${(data.rag.similarity * 100).toFixed(1)}%</div>
                        <div class="rag-text">${data.rag.closest_text.substring(0, 250)}...</div>
                    </div>
                </div>
            `;
            
            resultsDiv.style.display = 'block';
            resultsDiv.scrollIntoView({ behavior: 'smooth' });
        }
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if len(text) < 5:
            return jsonify({'error': 'Text too short'}), 400
        
        print(f"\nAnalyzing text: {text[:100]}...")
        
        # Run classification
        result = ensemble_classify([text])[0]
        print(f"Classification result: {result['final_label']} ({result['confidence']:.2f})")
        
        # Run RAG search
        rag_result = find_closest_event(text)
        print(f"RAG similarity: {rag_result['similarity']:.2f}")
        
        # Combine results
        result['rag'] = rag_result
        
        return jsonify(result)
    
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n*** Starting Propaganda Detection Web App ***")
    print("Open http://localhost:5000 in your browser\n")
    app.run(debug=True, host='0.0.0.0', port=5000)