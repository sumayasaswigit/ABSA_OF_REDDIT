# Aspect-Based Sentiment Analysis of Reddit Discussions

## 1. Project Overview

This research project focuses on **Aspect-Based Sentiment Analysis (ABSA)** of Reddit gaming discussions using Machine Learning and Deep Learning techniques.

The main objective is to identify aspects discussed in Reddit comments and classify the associated sentiment into three categories:

- Positive
- Negative
- Neutral

The project proposes an **Aspect-Weighted Feature Fusion (AWFF)** approach to improve sentiment classification by combining textual features with additional linguistic and sentiment-related features.

## 2. Problem Statement

Traditional sentiment analysis generally classifies an entire comment as positive, negative, or neutral. However, a single Reddit comment may discuss multiple aspects with different opinions.

This project addresses this limitation by identifying relevant aspects and using contextual, linguistic, and sentiment-based information to improve sentiment classification.

## 3. Objectives

- Perform sentiment analysis on Reddit gaming discussions.
- Identify relevant aspects from user comments.
- Extract meaningful textual and linguistic features.
- Develop an Aspect-Weighted Feature Fusion (AWFF) approach.
- Compare traditional Machine Learning and Deep Learning models.
- Evaluate baseline models against AWFF-enhanced models.

## 4. Dataset Description

The project uses a Reddit gaming comments dataset.

| Description | Details |
|---|---|
| Dataset file | `gaming_comments.csv` |
| Total records | 23,189 |
| Text column | Comment |
| Target column | sentiment |
| Number of classes | 3 |

### Sentiment Distribution

| Sentiment | Number of records |
|---|---:|
| Positive | 10,034 |
| Neutral | 9,083 |
| Negative | 4,072 |
| **Total** | **23,189** |

The dataset contains three sentiment categories: Positive, Neutral, and Negative.

## 5. Methodology

The project follows the following processing pipeline:

1. **Data Preprocessing**
   - Remove URLs and unnecessary characters.
   - Convert text into lowercase.
   - Clean and normalize Reddit comments.

2. **Aspect Extraction**
   - Use spaCy for Part-of-Speech tagging.
   - Extract noun phrases as candidate aspects.
   - Consider surrounding context for aspect-related information.

3. **Feature Extraction**
   - TF-IDF textual features.
   - VADER sentiment features.
   - Negation-related features.
   - Reddit linguistic features.
   - Additional numerical features.

4. **Aspect-Weighted Feature Fusion (AWFF)**

   The proposed AWFF approach combines TF-IDF textual representations with 17 numerical and linguistic features to provide additional information for sentiment classification.

5. **Model Training**

   Train and compare traditional Machine Learning and Deep Learning models.

6. **Model Evaluation**

   Evaluate model performance using accuracy, precision, recall, F1-score, and Macro-F1 score.

## 6. Algorithms Implemented

The project evaluates four algorithms using baseline and AWFF-enhanced approaches.

### Machine Learning Models
- Naive Bayes
- Logistic Regression

### Deep Learning Models
- Recurrent Neural Network (RNN)
- Long Short-Term Memory (LSTM)

## 7. Experimental Results

The following table compares baseline accuracy with AWFF-enhanced accuracy.

| Model | Baseline Accuracy | AWFF Accuracy | AWFF Macro-F1 |
|---|---:|---:|---:|
| Naive Bayes | 51.46% | 58.16% | 55.62% |
| Logistic Regression | 53.85% | 61.27% | 58.24% |
| RNN | 49.75% | 67.63% | 65.08% |
| LSTM | 53.32% | **68.51%** | **65.73%** |

### Best Performing Model

The LSTM model with Aspect-Weighted Feature Fusion achieved the best overall accuracy.

| Evaluation Metric | Score |
|---|---:|
| Accuracy | 68.51% |
| Precision | 68.59% |
| Recall | 68.51% |
| F1-Score | 68.55% |
| Macro-F1 | 65.73% |

The experimental results indicate that AWFF-enhanced models outperform their corresponding baseline models in accuracy.

## 8. Technologies Used

- Python
- Machine Learning
- Deep Learning
- TensorFlow / Keras
- Scikit-learn
- Pandas
- NumPy
- spaCy
- VADER Sentiment Analysis
- TF-IDF

## 9. Project Structure

```text
ABSA_OF_REDDIT/
│
├── models/
│   ├── awff_numeric_columns.pkl
│   ├── awff_scaler.pkl
│   ├── label_encoder.pkl
│   ├── logistic_regression_awff.pkl
│   ├── lstm_awff_model.keras
│   ├── lstm_tf-idf_baseline_model.keras
│   ├── naive_bayes_awff.pkl
│   ├── rnn_awff_model.keras
│   ├── rnn_tf-idf_baseline_model.keras
│   ├── tfidf_vectorizer.pkl
│   └── tokenizer.pkl
│
├── absa_model_training.py
├── absa_model_training_v2.py
├── gaming_comments.csv
├── .gitignore
└── README.md
```

## 10. Model Training and Execution

### Step 1: Clone the repository

```bash
git clone https://github.com/sumayaswigit/ABSA_OF_REDDIT.git
```

### Step 2: Navigate to the project directory

```bash
cd ABSA_OF_REDDIT
```

### Step 3: Create a virtual environment

```bash
python -m venv .venv
```

### Step 4: Activate the virtual environment

For Windows:

```bash
.venv\Scripts\activate
```

### Step 5: Install the required libraries

Install the dependencies used by the project:

```bash
pip install pandas numpy scikit-learn tensorflow spacy vaderSentiment
```

### Step 6: Run the training scripts

```bash
python absa_model_training.py
```

```bash
python absa_model_training_v2.py
```

Note: Use the script corresponding to the experiment you want to reproduce. Check the script configuration and required spaCy language model before execution.

## 11. Research Contribution

The main contribution of this research is the implementation and evaluation of an Aspect-Weighted Feature Fusion approach that combines textual representations with additional linguistic and sentiment-based features.

The experiments compare traditional Machine Learning and Deep Learning algorithms under baseline and AWFF-enhanced configurations.

## 12. Future Enhancements

- Improve aspect-level sentiment annotation.
- Explore advanced contextual language models such as BERT.
- Improve model performance through hyperparameter tuning.
- Evaluate the approach on additional Reddit datasets.
- Investigate improved aspect extraction techniques.

## 13. Repository

**GitHub:** https://github.com/sumayaswigit/ABSA_OF_REDDIT

---

**Project:** Aspect-Based Sentiment Analysis of Reddit Discussions

**Proposed Approach:** Aspect-Weighted Feature Fusion (AWFF)