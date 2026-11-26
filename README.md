# Aspect-Based Sentiment Analysis of Reddit Discussions Using Aspect-Weighted Feature Fusion (AWFF)

## 📌 Project Overview

This project focuses on performing Aspect-Based Sentiment Analysis (ABSA) on Reddit discussions using Natural Language Processing (NLP) and Machine Learning techniques.

Unlike traditional sentiment analysis, which assigns a single sentiment to an entire comment, ABSA identifies individual aspects and determines the sentiment associated with each aspect.

The project proposes an **Aspect-Weighted Feature Fusion (AWFF)** approach that combines multiple textual, linguistic, and contextual features to improve aspect-level sentiment classification.

The system explores traditional Machine Learning and Deep Learning models to classify sentiments into three categories:

- Positive
- Negative
- Neutral

## 🎯 Objectives

- Analyze Reddit discussions using NLP techniques.
- Extract meaningful aspects from user comments.
- Identify sentiment polarity associated with individual aspects.
- Develop an Aspect-Weighted Feature Fusion (AWFF) framework.
- Combine TF-IDF, aspect proximity, sentiment intensity, negation, contextual, and Reddit-specific features.
- Implement and compare baseline and proposed classification models.
- Evaluate model performance using standard classification metrics.

## ⚙️ Proposed Methodology

The project follows these major stages:

1. **Dataset Input:** Load Reddit discussion data from the CSV dataset.
2. **Text Preprocessing:** Clean and normalize textual data.
3. **Aspect Extraction:** Identify important aspects using NLP techniques.
4. **Feature Extraction:** Extract TF-IDF and linguistic features.
5. **AWFF Feature Fusion:** Combine multiple feature groups into a unified representation.
6. **Model Training:** Train baseline and proposed models.
7. **Evaluation:** Measure model performance using classification metrics.
8. **Visualization:** Analyze model performance and sentiment predictions.

## 🧠 Aspect-Weighted Feature Fusion (AWFF)

The proposed AWFF framework combines six feature groups:

| Feature | Description |
|---|---|
| TF-IDF | Identifies important terms in the comments. |
| Aspect Proximity | Captures the relationship between aspects and nearby sentiment words. |
| Sentiment Intensity | Measures the strength of sentiment expressions. |
| Negation Handling | Identifies negation words such as not, never, and no. |
| Contextual Features | Captures surrounding words and linguistic context. |
| Reddit-Specific Features | Handles slang, emojis, abbreviations, and informal expressions. |

## 🤖 Models Implemented

### Baseline Models
- Multinomial Naive Bayes
- Logistic Regression

### Proposed Models
- Recurrent Neural Network (RNN)
- Long Short-Term Memory (LSTM)

## 📊 Evaluation Metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Macro F1-Score
- Confusion Matrix

## 🛠️ Technologies Used

- **Programming Language:** Python
- **NLP:** spaCy, NLTK
- **Feature Extraction:** TF-IDF
- **Sentiment Analysis:** VADER, TextBlob
- **Machine Learning:** Scikit-learn
- **Deep Learning:** TensorFlow / Keras
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Development Environment:** Visual Studio Code
- **Version Control:** Git and GitHub

## 📂 Project Structure

```text
ABSA_Project/
│
├── .venv/                       # Python virtual environment (local)
│
├── models/                      # Saved trained models
│
├── results/                     # Model outputs and evaluation results
│
├── .gitignore                   # Files excluded from Git tracking
│
├── absa_model_training.py       # Initial model training implementation
│
├── absa_model_training_v2.py    # Updated model training implementation
│
├── gaming_comments.csv          # Reddit comments dataset
│
└── README.md                    # Project documentation
```

## 🚀 Installation and Execution

### 1. Clone the Repository

```bash
git clone https://github.com/sumayasaswigit/ABSA_OF_REDDIT.git
```

### 2. Navigate to the Project Directory

```bash
cd ABSA_OF_REDDIT
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate the Environment

For Windows:

```bash
.venv\Scripts\activate
```

### 5. Install Required Libraries

Install the dependencies required by the training scripts.

```bash
pip install pandas numpy scikit-learn matplotlib seaborn nltk spacy vaderSentiment textblob tensorflow
```

Additional NLP resources or spaCy language models may need to be installed depending on the implementation.

### 6. Run the Project

To execute the updated training implementation:

```bash
python absa_model_training_v2.py
```

To execute the initial implementation:

```bash
python absa_model_training.py
```

**Note:** The dataset file `gaming_comments.csv` should be available in the project directory when running the scripts. Check the script configuration if a different dataset path is used.

## 📁 Output

The project uses the following directories:

- **models/** – Stores trained model files and related artifacts.
- **results/** – Stores generated evaluation results, predictions, and visualizations.

The exact output files depend on the training script configuration.

## 🔬 Research Contribution

The project explores an aspect-aware feature fusion framework for sentiment analysis of Reddit discussions.

By combining lexical, aspect-related, sentiment, negation, contextual, and Reddit-specific features, the proposed AWFF approach aims to provide a richer representation for aspect-level sentiment classification.

## ⚠️ Dataset and Annotation Considerations

Aspect-level sentiment labels generated automatically using sentiment lexicons or rule-based methods should be treated as weakly supervised labels rather than human-annotated ground truth.

Reliable aspect-level evaluation requires manually verified reference annotations.

## 🔮 Future Scope

- Incorporating transformer-based models such as BERT and RoBERTa.
- Improving sarcasm and implicit aspect detection.
- Exploring advanced aspect extraction techniques.
- Evaluating the framework on additional Reddit datasets.
- Developing an interactive dashboard for sentiment visualization.

## 👩‍💻 Project Information

**Project Title:** Aspect-Based Sentiment Analysis of Reddit Discussions Using Aspect-Weighted Feature Fusion (AWFF)

**Domain:** Natural Language Processing and Machine Learning

**Institution:** KL University

**Project Type:** Academic Capstone Project

---
