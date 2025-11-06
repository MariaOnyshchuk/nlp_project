# NLP Propaganda Detection Model

A project focused on detecting and classifying text into propaganda/misinformation/oppozition classes using Natural Language Processing techniques.

## Project Description

This project implements ML&DL models to detect propaganda in text data using both TF-IDF and advanced sentence transformer embeddings. It includes comprehensive data preprocessing, feature engineering, and multiple classification approaches. The project provides both cleaned and lemmatized versions of training, validation, and testing datasets, with detailed model performance analysis and feature importance interpretations.
## Dataset

The dataset includes:
- **Training Data**:
  - `train_tg.csv` - Raw training data in telegram format
  - `train_cleaned.csv` - Cleaned training samples without capital letter / stopwords
  - `train_lemmatized.csv` - Lemmatized training samples for NLP processing
  
- **Test Data** and **Validation Data** have the same structure 

All datasets are in CSV format with preprocessed text features ready for model training.

## Installation & Setup

### Requirements
- Python 3.7+
- Jupyter Notebook
- Required libraries (install via pip):
  - pandas
  - numpy
  - scikit-learn
  - nltk (Natural Language Toolkit)
  - matplotlib
  - seaborn
  - lightgbm
  - sentence-transformers
  - pymorphy3
  - wordcloud

### Installation Steps

1. Clone the repository:
```bash
git clone https://github.com/MariaOnyshchuk/nlp_project.git
cd nlp_project
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required dependencies:
```bash
pip install pandas numpy scikit-learn nltk matplotlib seaborn jupyter lightgbm sentence-transformers pymorphy3 wordcloud```

## How to Run

1. **Start Jupyter Notebook**:
```bash
jupyter notebook
```

2. **Open the main notebook**:
   - Open `propaganda_detection_model.ipynb` in your browser

3. **Run the notebook cells** sequentially to:
   - Load and explore the dataset
   - Preprocess and clean text data
   - Perform feature engineering
   - Train machine learning models
   - Evaluate model performance
   - Generate analysis visualizations
BUT! Be aware that running the trainer for pretrained models can take up to 3 hours each. 

### Using the Dataset

The cleaned and lemmatized datasets are ready to use:
- For traditional ML models, use the cleaned or lemmatized CSV files
- The models include train/val splits for validation
- Use the test set (`test_lemmatized.csv`) for final model evaluation

## Project Workflow

1. **Data Loading & Exploration** - Load datasets and analyze their characteristics
2. **Data Preprocessing** - Text cleaning and normalization
3. **Lemmatization** - Processing lemmatized versions for NLP tasks
4. **Feature Engineering** - Extract relevant features for classification
5. **Model Training** - Train multiple machine learning models
6. **Model Evaluation** - Test and compare model performance
7. **Visualization** - Generate insights and performance metrics

## Authors/Contributors

- **Maria Onyshchuk** - [@MariaOnyshchuk](https://github.com/MariaOnyshchuk)
- **Dmytro Malyk** - [@DmytroMalyk](https://github.com/DmytroMalyk)
- **Roman Pavlosiuk** - [@RomanPavlosiuk](https://github.com/gllekkoff)

## License

This project is provided as-is for educational and research purposes.

## Repository Structure

```
nlp_project/
├── data/
│   ├── train_cleaned.csv
│   ├── train_lemmatized.csv
│   ├── train_tg.csv
│   ├── test_cleaned.csv
│   ├── test_lemmatized.csv
│   ├── test_tg.csv
│   └── val_lemmatized.csv
├── propaganda_detection_model.ipynb
└── README.md
```


## Model Performance

The project implements multiple machine learning and deep learning models with different embedding approaches.

### ML Models with TF-IDF Embeddings

| Model | Embedding | F1-Score | Notes |
|-------|-----------|----------|-------|
| Naive Bayes | TF-IDF (max_features=5000) | 0.68 | Fast baseline model |
| Random Forest | TF-IDF (max_features=5000) | 0.69 | 200 estimators |
| SVM | TF-IDF (max_features=5000) | 0.69 | Linear kernel |
| LightGBM | TF-IDF (max_features=5000) | 0.70 | 500 estimators, gradient boosting |
| Logistic Regression | TF-IDF n-grams (1-3, max=10000) | 0.72 | Best TF-IDF performer |

### Advanced Embeddings

| Model | Embedding | F1-Score | Notes |
|-------|-----------|----------|-------|
| Logistic Regression | SentenceTransformers (all-MiniLM-L6-v2) | 0.54 | Lower performance, suggests need for domain-specific embeddings |

### Key Findings

**Best Performing Model**: Logistic Regression with TF-IDF n-grams (1-3) - F1: 0.72

**Feature Importance by Class**:
- **Misinformation (Class 0)**: "донбасс" (Donbass), "днр" (DNR), "лнр" (LNR), "решать" (resolve)
- **Propaganda (Class 1)**: "сообщить" (reported), "заявить" (stated), "соловей" (Solovyov), "газета" (newspaper)
- **Opposition (Class 2)**: "невзоров" (Nevzorov), "петербург" (Petersburg), "белковский" (Belkovsky), "задержать" (arrest)


## Notes

- The project includes multiple data preprocessing approaches (cleaned, lemmatized, and telegram format)
- All datasets are pre-processed and ready for model training
- The Jupyter notebook contains comprehensive data analysis and multiple ML model implementations
- Latest updates include lemmatized validation set for improved model evaluation
