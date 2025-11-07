import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')

def train_model():
    print("Loading data...")
    train = pd.read_csv('../data/train_lemmatized.csv')
    test = pd.read_csv('../data/test_lemmatized.csv')

    train = train.dropna(subset=['message_lemmatized', 'label'])
    test = test.dropna(subset=['message_lemmatized', 'label'])

    print(f"Train size: {len(train)}, Test size: {len(test)}")

    X_train = train['message_lemmatized']
    y_train = train['label'].values
    X_test = test['message_lemmatized']
    y_test = test['label'].values

    print("\nTraining TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=30000,
        lowercase=True
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("Training Logistic Regression model...")
    clf = LogisticRegression(
        max_iter=1000,
        random_state=42,
        n_jobs=-1,
        solver='saga',
        multi_class='multinomial'
    )

    clf.fit(X_train_tfidf, y_train)

    print("\nEvaluating model...")
    y_pred = clf.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"F1 (macro): {f1:.4f}")
    print(f"Precision: {report['macro avg']['precision']:.4f}")
    print(f"Recall:    {report['macro avg']['recall']:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                                target_names=['Misinformation', 'Propaganda', 'Opposition'],
                                zero_division=0))

    print("\nConfusion Matrix:")
    print(cm)

    results = {
        'method': 'TF-IDF (trigrams 1-3)',
        'classifier': 'LR',
        'accuracy': accuracy,
        'f1_macro': f1,
        'precision_macro': report['macro avg']['precision'],
        'recall_macro': report['macro avg']['recall'],
    }
    results_df = pd.DataFrame([results])
    results_df.to_csv('model_results.csv', index=False)
    print("\nResults saved to 'model_results.csv'")

    os.makedirs('model', exist_ok=True)

    print("\nSaving model and vectorizer to ./model/ directory...")
    with open('model/propaganda_model.pkl', 'wb') as f:
        pickle.dump(clf, f)

    with open('model/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

    label_map = {0: 'Misinformation', 1: 'Propaganda', 2: 'Opposition'}
    with open('model/label_map.pkl', 'wb') as f:
        pickle.dump(label_map, f)

    print("\nModel saved as 'model/propaganda_model.pkl'")
    print("Vectorizer saved as 'model/vectorizer.pkl'")
    print("Label map saved as 'model/label_map.pkl'")

if __name__ == "__main__":
    train_model()
