# NLP Propaganda Detection Model

A machine learning project focused on detecting and analyzing propaganda in text using Natural Language Processing techniques.

## Project Description

This project implements machine learning models to detect propaganda in text data. It includes comprehensive data preprocessing, feature engineering, and multiple machine learning approaches including traditional ML models and deep learning techniques. The project provides both cleaned and lemmatized versions of training and testing datasets for flexible modeling approaches.

## Dataset

The dataset includes:
- **Training Data**: 
  - `train_cleaned.csv` - Cleaned training samples
  - `train_lemmatized.csv` - Lemmatized training samples for NLP processing
  - `train_tg.csv` - Training data in telegram format
  
- **Test Data**: 
  - `test_cleaned.csv` - Cleaned test samples
  - `test_lemmatized.csv` - Lemmatized test samples
  - `test_tg.csv` - Test data in telegram format

- **Validation Data**:
  - `val_lemmatized.csv` - Lemmatized validation set for model evaluation

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
pip install pandas numpy scikit-learn nltk matplotlib seaborn jupyter
```

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

### Using the Dataset

The cleaned and lemmatized datasets are ready to use:
- For traditional ML models, use the cleaned or lemmatized CSV files
- The models include train/test splits for validation
- Use the validation set (`val_lemmatized.csv`) for final model evaluation

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

## Notes

- The project includes multiple data preprocessing approaches (cleaned, lemmatized, and telegram format)
- All datasets are pre-processed and ready for model training
- The Jupyter notebook contains comprehensive data analysis and multiple ML model implementations
- Latest updates include lemmatized validation set for improved model evaluation
