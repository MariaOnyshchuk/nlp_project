import pickle
import warnings
warnings.filterwarnings('ignore')

class PropagandaDetector:
    def __init__(self):
        print("Loading model...")
        with open('model/propaganda_model.pkl', 'rb') as f:
            self.model = pickle.load(f)

        with open('model/vectorizer.pkl', 'rb') as f:
            self.vectorizer = pickle.load(f)

        with open('model/label_map.pkl', 'rb') as f:
            self.label_map = pickle.load(f)

        print("Model loaded successfully!\n")
    
    def predict(self, text):
        text_tfidf = self.vectorizer.transform([text])

        prediction = self.model.predict(text_tfidf)[0]
        probabilities = self.model.predict_proba(text_tfidf)[0]

        label = self.label_map[prediction]

        return label, probabilities

    def display_results(self, text, label, probs):
        print("\n" + "="*70)
        print("ANALYSIS RESULTS")
        print("="*70)
        print(f"\nInput text: {text[:100]}{'...' if len(text) > 100 else ''}")
        print(f"\nPredicted class: {label}")
        print("\nConfidence scores:")
        for idx, class_name in self.label_map.items():
            confidence = probs[idx] * 100
            bar_length = int(confidence / 2)
            bar = "█" * bar_length
            print(f"  {class_name:15s} [{bar:50s}] {confidence:5.2f}%")
        print("="*70 + "\n")
def main():
    print("="*70)
    print("RUSSIAN NEWS PROPAGANDA DETECTION SYSTEM")
    print("="*70)
    print("\nThis system classifies Russian news text into three categories:")
    print("  • Misinformation - False or misleading information")
    print("  • Propaganda - Biased information promoting a viewpoint")
    print("  • Opposition - Counter-narrative or opposing viewpoints")
    print()

    try:
        detector = PropagandaDetector()
    except FileNotFoundError:
        print("ERROR: Model files not found!")
        print("Please run 'python train_model.py' first to train the model.")
        return

    print("Enter text to analyze (or 'quit' to exit)")
    print("-" * 70)

    while True:
        print("\n> ", end="")
        text = input().strip()

        if text.lower() in ['quit', 'exit', 'q']:
            print("\nThank you for using the propaganda detection system!")
            break

        if not text:
            print("Please enter some text.")
            continue

        label, probs = detector.predict(text)
        detector.display_results(text, label, probs)

if __name__ == "__main__":
    main()
