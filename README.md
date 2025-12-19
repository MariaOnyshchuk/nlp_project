***

# NLP Propaganda Detection Model

A project for detecting and classifying text into **propaganda**, misinformation, and opposition classes using classical NLP pipelines and modern transformer-based methods, including a Retrieval-Augmented Generation (RAG) component. 

## Project Description

This project implements both machine learning and deep learning models to detect propaganda in text using TF‑IDF features, traditional classifiers, and sentence-transformer embeddings.  It also includes an ensemble that combines a RAG-based model with a logistic regression classifier to improve robustness and interpretability.  The workflow covers data preprocessing, feature engineering, model training, evaluation, and visualization of results. 

## RAG Component

The RAG module retrieves semantically similar training samples or reference snippets using sentence-transformer embeddings and passes them, together with the input text, to a generative model that outputs a propaganda-related label.  These RAG predictions are then ensembled with a logistic regression model trained on classical features, forming the “RAG + LR” ensemble used in the later notebooks.  This design lets the system both ground its decisions in real examples and leverage learned decision boundaries from traditional models. 

## Dataset

The `data/` folder contains preprocessed CSV files for train, validation, and test splits in several variants. 

- Training data:
  - `train_tg.csv` – raw training data in Telegram-like format. 
  - `train_cleaned.csv` – cleaned text (lowercased, no stopwords, basic normalization). 
  - `train_lemmatized.csv` – lemmatized text for downstream NLP models. 
- Validation and test data:
  - `test_tg.csv`, `test_cleaned.csv`, `test_lemmatized.csv`, `val_lemmatized.csv` – same structure as training files, ready for evaluation. 

All datasets are in CSV format and are ready to be fed directly into the notebooks. 

## Repository Structure

```text
nlp_project/
├── data/
│   ├── train_cleaned.csv
│   ├── train_lemmatized.csv
│   ├── train_tg.csv
│   ├── test_cleaned.csv
│   ├── test_lemmatized.csv
│   ├── test_tg.csv
│   └── val_lemmatized.csv
├── demo/
│   ├── ...
├── models/
│   ├── ...
├── plots/
│   ├── ...
├── data_preprocessing.ipynb
├── plotting.ipynb
├── propaganda_detection.ipynb
├── propaganda_ensemble.ipynb
├── propaganda_rag_lr_ensemble_comparison.ipynb
├── shap-analysis.ipynb
├── requirements.txt
└── README.md
```


- `data_preprocessing.ipynb`: text cleaning, lemmatization, and dataset preparation. 
- `propaganda_detection.ipynb`: baseline models with TF‑IDF and classical classifiers. 
- `propaganda_ensemble.ipynb`: ensemble logic, including RAG + LR combination. 
- `propaganda_rag_lr_ensemble_comparison.ipynb`: comparison of RAG, LR, and ensemble performance with plots. 
- `plotting.ipynb` and `plots/`: scripts and outputs for visualizations. 
- `shap-analysis.ipynb`: SHAP-based feature importance analysis for interpretability. 

## Installation & Setup

### Requirements

- Python 3.7+  
- Jupyter Notebook  
- Dependencies from `requirements.txt` (pandas, numpy, scikit-learn, nltk, matplotlib, seaborn, lightgbm, sentence-transformers, pymorphy3, wordcloud, etc.). 

### Installation Steps

```bash
git clone https://github.com/MariaOnyshchuk/nlp_project.git
cd nlp_project

python -m venv venv
# Linux / macOS
source venv/bin/activate
# Windows
# venv\Scripts\activate

pip install -r requirements.txt
```


## How to Run the Notebooks

1. Start Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
2. Open `propaganda_detection.ipynb`. 
3. Run cells sequentially to:
   - Load and explore the dataset. 
   - Preprocess text and engineer features. 
   - Train and evaluate ML models. 
   - Optionally run RAG and ensemble experiments in the corresponding notebooks. 

Note: training some pretrained or RAG-based models can take up to several hours per run. 

## How to Run the Demo (Web UI)

The `demo/` folder contains a simple web interface to interact with the trained propaganda detection models. 

Typical workflow (adjust names if your main script differs):

1. Navigate to the demo directory:
   ```bash
   cd demo
   ```
2. Make sure the virtual environment from the project root is activated and all dependencies are installed. 
3. Run the demo app (for example, if it is a Streamlit app):
   ```bash
   streamlit run app.py
   ```
   or, if it is a plain Python web server:
   ```bash
   python app.py
   ```
4. Open the local URL shown in the terminal (usually `http://localhost:8501` for Streamlit or `http://127.0.0.1:8000` for many other frameworks).  
5. Enter a text sample into the interface to see predicted propaganda/misinformation/opposition labels, optionally with RAG-based explanations depending on the configuration.  

(If your demo script has a different filename or uses another framework, replace `app.py` and the command accordingly.)

## Project Workflow

- Data loading and initial exploratory analysis. 
- Text preprocessing and lemmatization using multiple pipelines. 
- Feature extraction with TF‑IDF, classical n‑grams, and sentence-transformer embeddings. 
- Model training with several algorithms (including lightGBM and logistic regression) and RAG. 
- Model evaluation, plotting metrics, and SHAP-based interpretation. 

## Authors

- [@MariaOnyshchuk](https://github.com/MariaOnyshchuk)  
- [@DmytroMalyk](https://github.com/DmytroMalyk)  
- [@gllekkoff](https://github.com/gllekkoff)  

## License

This project is provided as-is for educational and research purposes. 

***
