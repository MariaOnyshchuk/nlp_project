import pickle
import gradio as gr
import warnings
warnings.filterwarnings('ignore')

import re
import pymorphy3

morph = pymorphy3.MorphAnalyzer()

class PropagandaDetector:
    def __init__(self):
        with open('model/propaganda_model.pkl', 'rb') as f:
            self.model = pickle.load(f)

        with open('model/vectorizer.pkl', 'rb') as f:
            self.vectorizer = pickle.load(f)

        with open('model/label_map.pkl', 'rb') as f:
            self.label_map = pickle.load(f)

    def predict(self, text):
        if not text or not text.strip():
            return {
                "Misinformation": 0.0,
                "Propaganda": 0.0,
                "Opposition": 0.0
            }

        tokens = re.findall(r'[А-Яа-яA-Za-z]+', text.lower())
        lemmatized_tokens = [morph.parse(word)[0].normal_form for word in tokens]
        lemmatized_text = ' '.join(lemmatized_tokens)

        text_tfidf = self.vectorizer.transform([lemmatized_text])
        probabilities = self.model.predict_proba(text_tfidf)[0]

        result = {
            self.label_map[idx]: float(prob) 
            for idx, prob in enumerate(probabilities)
        }

        return result

try:
    detector = PropagandaDetector()
except FileNotFoundError:
    print("ERROR: Model files not found!")
    print("Please run 'python train_model.py' first.")
    exit(1)

examples = [
    ["Украинские войска намеренно обстреливают мирное население Донбасса"],
    ["Россия проводит специальную военную операцию для защиты русскоязычного населения"],
    ["Независимые наблюдатели сообщают о нарушениях прав человека с обеих сторон конфликта"],
]

with gr.Blocks(title="Propaganda Detection System", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🔍 Russian News Propaganda Detection System
    
    This system uses machine learning (Logistic Regression + TF-IDF) to classify Russian news text into three categories:
    - **Misinformation**: False or misleading information
    - **Propaganda**: Biased information promoting a specific viewpoint
    - **Opposition**: Counter-narrative or opposing viewpoints
    """)

    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="Enter Russian text to analyze",
                placeholder="Enter text...",
                lines=5
            )

            with gr.Row():
                submit_btn = gr.Button("Analyze", variant="primary", size="lg")
                clear_btn = gr.ClearButton([text_input], value="Clear")

        with gr.Column(scale=1):
            output = gr.Label(
                label="Classification Results",
                num_top_classes=3
            )

    gr.Markdown("### Example texts (click to use):")
    gr.Examples(
        examples=examples,
        inputs=text_input,
        label=""
    )

    gr.Markdown("""
    ---
    **Note**: This is a demo system trained on lemmatized Russian news articles. 
    Results should be interpreted as indicators rather than definitive classifications.
    """)

    submit_btn.click(
        fn=detector.predict,
        inputs=text_input,
        outputs=output
    )
    
    text_input.submit(
        fn=detector.predict,
        inputs=text_input,
        outputs=output
    )

if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
