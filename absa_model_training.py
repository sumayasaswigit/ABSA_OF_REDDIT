# ============================================================
# ABSA MODEL TRAINING WITH BASELINE + FULL AWFF
#
# Project:
#   Aspect-Based Sentiment Analysis of Reddit Discussions
#
# Models:
#   1. Naive Bayes
#   2. Logistic Regression
#   3. RNN
#   4. LSTM
#
# Experiments:
#   A. TF-IDF baseline
#   B. Proposed AWFF
#
# Dataset:
#   gaming_comments.csv
#
# IMPORTANT RESEARCH NOTE:
#   The dataset contains comment-level sentiment labels, not
#   human-annotated aspect-level labels. When a comment contains
#   multiple extracted aspects, its comment-level label is used
#   for the extracted aspect record. This is weak/distant
#   supervision and must be reported as a dataset limitation.
#
# AWFF FEATURES:
#   1. TF-IDF
#   2. Aspect proximity
#   3. Sentiment intensity
#   4. Negation
#   5. Contextual information
#   6. Reddit-specific linguistic features
#
# The original CSV is never modified.
# ============================================================

import os
import re
import random
import warnings
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
import tensorflow as tf

from scipy.sparse import hstack, csr_matrix

from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Embedding,
    SimpleRNN,
    LSTM,
    Dense,
    Dropout,
    Concatenate,
)
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings("ignore")


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42

DATASET_PATH = "gaming_comments.csv"

RESULTS_DIR = "results"
MODELS_DIR = "models"

TEST_SIZE = 0.20

TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.95

MAX_WORDS = 15000
MAX_SEQUENCE_LENGTH = 120

EMBEDDING_DIM = 128
RECURRENT_UNITS = 96

EPOCHS = 15
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.10
EARLY_STOPPING_PATIENCE = 3

CONTEXT_WINDOW = 8


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


# ============================================================
# 3. NLTK / VADER
# ============================================================

print("\nChecking VADER lexicon...")

try:
    sia = SentimentIntensityAnalyzer()
except LookupError:
    print("Downloading VADER lexicon...")
    nltk.download("vader_lexicon", quiet=True)
    sia = SentimentIntensityAnalyzer()


# ============================================================
# 4. LOAD SPACY
# ============================================================

print("\nLoading spaCy model...")

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise OSError(
        "\nspaCy English model is missing.\n"
        "Run:\n"
        "python -m spacy download en_core_web_sm\n"
    )


# ============================================================
# 5. LOAD DATASET
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(DATASET_PATH)

print("Dataset shape:", df.shape)
print("Columns:", list(df.columns))

if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

required_columns = ["Comment", "sentiment"]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(
            f"Required column '{column}' was not found."
        )

df = df.dropna(subset=required_columns).copy()

df["Comment"] = df["Comment"].astype(str)
df["sentiment"] = df["sentiment"].astype(str).str.strip()

print("\nOriginal sentiment distribution:")
print(df["sentiment"].value_counts())


# ============================================================
# 6. TEXT PREPROCESSING
# ============================================================

def clean_text(text):
    """
    Cleaning used for lexical model input.

    The raw comment is retained separately because Reddit-specific
    linguistic features are extracted from the original text.
    """

    text = str(text)

    # Remove URLs.
    text = re.sub(
        r"http\S+|www\S+|https\S+",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    # Remove Reddit user references.
    text = re.sub(r"/u/\w+", " ", text)
    text = re.sub(r"\bu/\w+", " ", text)

    # Remove subreddit references.
    text = re.sub(r"/r/\w+", " ", text)
    text = re.sub(r"\br/\w+", " ", text)

    # Keep letters, numbers and common punctuation.
    text = re.sub(
        r"[^A-Za-z0-9!?.,'%\s]",
        " ",
        text,
    )

    text = re.sub(r"\s+", " ", text)

    return text.strip().lower()


df["clean_text"] = df["Comment"].apply(clean_text)

df = df[df["clean_text"].str.len() > 0].copy()

print("\nAfter preprocessing:", df.shape)


# ============================================================
# 7. ASPECT EXTRACTION
# ============================================================

def extract_aspects_from_doc(doc):
    """
    Automatic aspect candidate extraction using:
      - noun phrases
      - POS tagging

    No manual aspect labels are added to the CSV.
    """

    aspects = []

    # Noun phrases.
    for chunk in doc.noun_chunks:
        aspect = chunk.text.strip().lower()

        if not aspect:
            continue

        if any(token.pos_ == "PRON" for token in chunk):
            continue

        if not any(
            token.pos_ in {"NOUN", "PROPN"}
            for token in chunk
        ):
            continue

        alpha_count = sum(char.isalpha() for char in aspect)

        if alpha_count < 2:
            continue

        if aspect.replace(" ", "").isdigit():
            continue

        aspects.append(
            {
                "aspect": aspect,
                "start": chunk.start,
                "end": chunk.end,
            }
        )

    # Single nouns/proper nouns not already represented.
    existing_spans = {
        (item["start"], item["end"])
        for item in aspects
    }

    for token in doc:
        if token.pos_ not in {"NOUN", "PROPN"}:
            continue

        if token.is_stop:
            continue

        if len(token.text.strip()) < 2:
            continue

        span = (token.i, token.i + 1)

        if span in existing_spans:
            continue

        aspect = token.text.strip().lower()

        alpha_count = sum(char.isalpha() for char in aspect)

        if alpha_count < 2:
            continue

        aspects.append(
            {
                "aspect": aspect,
                "start": token.i,
                "end": token.i + 1,
            }
        )

    # Deduplicate spans.
    unique = []
    seen = set()

    for item in aspects:
        key = (
            item["aspect"],
            item["start"],
            item["end"],
        )

        if key not in seen:
            unique.append(item)
            seen.add(key)

    return unique


# ============================================================
# 8. ASPECT-LEVEL FEATURE EXTRACTION
# ============================================================

NEGATION_WORDS = {
    "not",
    "no",
    "never",
    "neither",
    "nor",
    "n't",
    "cannot",
    "can't",
    "don't",
    "doesn't",
    "didn't",
    "isn't",
    "wasn't",
    "weren't",
    "won't",
    "wouldn't",
    "couldn't",
    "shouldn't",
}


def count_repeated_characters(text):
    """
    Number of alphabetic sequences containing a character repeated
    at least three times, e.g. 'soooo', 'coool'.
    """
    return len(
        re.findall(r"(.)\1{2,}", text.lower())
    )


def count_exclamations(text):
    return text.count("!")


def count_question_marks(text):
    return text.count("?")


def count_uppercase_words(text):
    words = re.findall(r"\b[A-Za-z]{2,}\b", text)
    return sum(word.isupper() for word in words)


def count_emoticon_patterns(text):
    patterns = [
        r":\)",
        r":\(",
        r";\)",
        r":D",
        r"<3",
        r":/",
        r":-",
    ]

    return sum(
        len(re.findall(pattern, text))
        for pattern in patterns
    )


def get_reddit_features(raw_comment):
    """
    Reddit-specific linguistic signals.

    These are kept as numeric AWFF features rather than being
    used as the target.
    """

    text = str(raw_comment)

    word_count = len(
        re.findall(r"\b\w+\b", text)
    )

    char_count = len(text)

    uppercase_ratio = 0.0

    letters = [
        char for char in text
        if char.isalpha()
    ]

    if letters:
        uppercase_ratio = sum(
            char.isupper()
            for char in letters
        ) / len(letters)

    return {
        "reddit_word_count": word_count,
        "reddit_char_count": char_count,
        "reddit_exclamation_count": count_exclamations(text),
        "reddit_question_count": count_question_marks(text),
        "reddit_uppercase_ratio": uppercase_ratio,
        "reddit_repeated_char_count": count_repeated_characters(text),
        "reddit_emoticon_count": count_emoticon_patterns(text),
        "reddit_uppercase_word_count": count_uppercase_words(text),
    }


print("\nExtracting aspects and AWFF features...")

texts = df["clean_text"].tolist()
raw_comments = df["Comment"].tolist()
sentiments = df["sentiment"].tolist()
comment_ids = df.index.tolist()

aspect_records = []

for count, (
    comment_id,
    cleaned_text,
    raw_comment,
    sentiment,
    doc,
) in enumerate(
    zip(
        comment_ids,
        texts,
        raw_comments,
        sentiments,
        nlp.pipe(
            texts,
            batch_size=256,
        ),
    ),
    start=1,
):
    aspects = extract_aspects_from_doc(doc)

    reddit_features = get_reddit_features(raw_comment)

    for item in aspects:
        aspect = item["aspect"]
        start = item["start"]
        end = item["end"]

        # --------------------------------------------
        # Aspect-centered context
        # --------------------------------------------

        context_start = max(
            0,
            start - CONTEXT_WINDOW,
        )

        context_end = min(
            len(doc),
            end + CONTEXT_WINDOW,
        )

        context_tokens = doc[
            context_start:context_end
        ]

        context = context_tokens.text.strip()

        # --------------------------------------------
        # Aspect proximity
        # --------------------------------------------

        if len(doc) > 1:
            aspect_center = (
                (start + end - 1) / 2
            )

            distance_from_center = abs(
                aspect_center -
                ((len(doc) - 1) / 2)
            )

            normalized_distance = (
                distance_from_center /
                max((len(doc) - 1) / 2, 1.0)
            )

            aspect_proximity = (
                1.0 - normalized_distance
            )
        else:
            aspect_proximity = 1.0

        # --------------------------------------------
        # Sentiment intensity
        # --------------------------------------------

        vader_scores = sia.polarity_scores(
            context
        )

        sentiment_compound = (
            vader_scores["compound"]
        )

        sentiment_positive = (
            vader_scores["pos"]
        )

        sentiment_negative = (
            vader_scores["neg"]
        )

        sentiment_neutral = (
            vader_scores["neu"]
        )

        # --------------------------------------------
        # Negation
        # --------------------------------------------

        negation_count = sum(
            1
            for token in context_tokens
            if token.text.lower()
            in NEGATION_WORDS
        )

        # --------------------------------------------
        # Contextual information
        # --------------------------------------------

        context_length = len(
            context_tokens
        )

        context_position = (
            (start + end - 1) /
            max(len(doc) - 1, 1)
        )

        aspect_token_length = end - start

        # --------------------------------------------
        # Reddit-specific linguistic features
        # --------------------------------------------

        record = {
            "comment_id": comment_id,
            "aspect": aspect,
            "context": context,
            "sentiment": sentiment,

            "aspect_proximity": aspect_proximity,

            "sentiment_compound": sentiment_compound,
            "sentiment_positive": sentiment_positive,
            "sentiment_negative": sentiment_negative,
            "sentiment_neutral": sentiment_neutral,

            "negation_count": negation_count,

            "context_length": context_length,
            "context_position": context_position,
            "aspect_token_length": aspect_token_length,
        }

        record.update(reddit_features)

        aspect_records.append(record)

    if count % 1000 == 0:
        print(
            f"Processed {count}/{len(texts)} comments..."
        )


aspect_df = pd.DataFrame(aspect_records)

if aspect_df.empty:
    raise ValueError(
        "No aspects were extracted. Check the dataset "
        "and spaCy model."
    )

aspect_df = aspect_df.dropna(
    subset=[
        "aspect",
        "context",
        "sentiment",
    ]
).copy()

aspect_df = aspect_df[
    aspect_df["aspect"].str.strip() != ""
].copy()

print(
    "\nExtracted aspect records:",
    len(aspect_df),
)

aspect_df.to_csv(
    os.path.join(
        RESULTS_DIR,
        "extracted_aspects.csv",
    ),
    index=False,
)


# ============================================================
# 9. LABEL ENCODING
# ============================================================

label_encoder = LabelEncoder()

aspect_df["label"] = label_encoder.fit_transform(
    aspect_df["sentiment"]
)

num_classes = len(
    label_encoder.classes_
)

print("\nSentiment classes:")

for index, label in enumerate(
    label_encoder.classes_
):
    print(f"{index} = {label}")

print("\nNumber of classes:", num_classes)


# ============================================================
# 10. CLASS DISTRIBUTION
# ============================================================

class_counts = aspect_df[
    "sentiment"
].value_counts()

plt.figure(figsize=(8, 5))

sns.barplot(
    x=class_counts.index,
    y=class_counts.values,
)

plt.title(
    "Aspect-Level Weakly Supervised Sentiment Distribution"
)
plt.xlabel("Sentiment")
plt.ylabel("Number of Aspect Records")
plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "sentiment_distribution.png",
    ),
    dpi=300,
)

plt.close()


# ============================================================
# 11. GROUPED TRAIN / TEST SPLIT
# ============================================================

print("\nCreating grouped train/test split...")

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=TEST_SIZE,
    random_state=SEED,
)

train_indices, test_indices = next(
    splitter.split(
        aspect_df,
        aspect_df["label"],
        groups=aspect_df["comment_id"],
    )
)

train_df = aspect_df.iloc[
    train_indices
].copy()

test_df = aspect_df.iloc[
    test_indices
].copy()

print("Training aspect records:", len(train_df))
print("Testing aspect records :", len(test_df))

print(
    "\nTrain comments:",
    train_df["comment_id"].nunique(),
)

print(
    "Test comments :",
    test_df["comment_id"].nunique(),
)


# ============================================================
# 12. TARGET LABELS
# ============================================================

y_train = train_df["label"].values
y_test = test_df["label"].values


# ============================================================
# 13. CLASS WEIGHTS
# ============================================================

classes = np.unique(y_train)

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train,
)

class_weights = {
    int(cls): float(weight)
    for cls, weight in zip(
        classes,
        class_weights_array,
    )
}

print("\nClass weights:")

for cls, weight in class_weights.items():
    label = label_encoder.inverse_transform(
        [cls]
    )[0]

    print(
        f"{label}: {weight:.4f}"
    )


# ============================================================
# 14. TF-IDF BASELINE FEATURES
# ============================================================

print("\nBuilding TF-IDF baseline features...")

tfidf = TfidfVectorizer(
    max_features=TFIDF_MAX_FEATURES,
    ngram_range=TFIDF_NGRAM_RANGE,
    min_df=TFIDF_MIN_DF,
    max_df=TFIDF_MAX_DF,
    sublinear_tf=True,
    strip_accents="unicode",
)

X_train_tfidf = tfidf.fit_transform(
    train_df["context"]
)

X_test_tfidf = tfidf.transform(
    test_df["context"]
)

print(
    "TF-IDF train shape:",
    X_train_tfidf.shape,
)

print(
    "TF-IDF test shape :",
    X_test_tfidf.shape,
)


# ============================================================
# 15. FULL AWFF NUMERIC FEATURES
# ============================================================

AWFF_NUMERIC_COLUMNS = [
    # Aspect-aware
    "aspect_proximity",

    # Sentiment intensity
    "sentiment_compound",
    "sentiment_positive",
    "sentiment_negative",
    "sentiment_neutral",

    # Negation
    "negation_count",

    # Contextual
    "context_length",
    "context_position",
    "aspect_token_length",

    # Reddit-specific
    "reddit_word_count",
    "reddit_char_count",
    "reddit_exclamation_count",
    "reddit_question_count",
    "reddit_uppercase_ratio",
    "reddit_repeated_char_count",
    "reddit_emoticon_count",
    "reddit_uppercase_word_count",
]

print(
    "\nAWFF numeric feature count:",
    len(AWFF_NUMERIC_COLUMNS),
)

scaler = MinMaxScaler()

X_train_numeric = scaler.fit_transform(
    train_df[AWFF_NUMERIC_COLUMNS]
)

X_test_numeric = scaler.transform(
    test_df[AWFF_NUMERIC_COLUMNS]
)

X_train_numeric_sparse = csr_matrix(
    X_train_numeric
)

X_test_numeric_sparse = csr_matrix(
    X_test_numeric
)

# Proposed AWFF = lexical TF-IDF + all numeric AWFF features.
X_train_awff = hstack(
    [
        X_train_tfidf,
        X_train_numeric_sparse,
    ]
).tocsr()

X_test_awff = hstack(
    [
        X_test_tfidf,
        X_test_numeric_sparse,
    ]
).tocsr()

print(
    "AWFF train shape:",
    X_train_awff.shape,
)

print(
    "AWFF test shape :",
    X_test_awff.shape,
)


# ============================================================
# 16. EVALUATION
# ============================================================

all_results = []


def evaluate_model(
    model_name,
    feature_set,
    y_true,
    y_pred,
):
    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        )
    )

    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )
    )

    print("\n" + "=" * 75)
    print(f"{model_name} | {feature_set}")
    print("=" * 75)

    print(f"Accuracy        : {accuracy:.4f}")
    print(f"Weighted Prec.  : {precision:.4f}")
    print(f"Weighted Recall : {recall:.4f}")
    print(f"Weighted F1     : {f1:.4f}")
    print(f"Macro Precision : {macro_precision:.4f}")
    print(f"Macro Recall    : {macro_recall:.4f}")
    print(f"Macro F1        : {macro_f1:.4f}")

    print("\nClassification Report:")

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=label_encoder.classes_,
            zero_division=0,
        )
    )

    all_results.append(
        {
            "Model": model_name,
            "Feature_Set": feature_set,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Macro_F1": macro_f1,
        }
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
    )

    plt.figure(figsize=(7, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_,
        cmap="Blues",
    )

    plt.title(
        f"{model_name} - {feature_set} - Confusion Matrix"
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    safe_name = (
        f"{model_name}_{feature_set}"
        .replace(" ", "_")
        .replace("/", "_")
    )

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            f"{safe_name}_confusion_matrix.png",
        ),
        dpi=300,
    )

    plt.close()


# ============================================================
# 17. NAIVE BAYES — TF-IDF BASELINE
# ============================================================

print("\nTraining Naive Bayes — TF-IDF baseline...")

nb_baseline = MultinomialNB()

nb_sample_weights = np.array(
    [
        class_weights[int(label)]
        for label in y_train
    ]
)

nb_baseline.fit(
    X_train_tfidf,
    y_train,
    sample_weight=nb_sample_weights,
)

nb_baseline_pred = nb_baseline.predict(
    X_test_tfidf
)

evaluate_model(
    "Naive Bayes",
    "TF-IDF Baseline",
    y_test,
    nb_baseline_pred,
)


# ============================================================
# 18. NAIVE BAYES — AWFF
# ============================================================

print("\nTraining Naive Bayes — AWFF...")

nb_awff = MultinomialNB()

nb_awff.fit(
    X_train_awff,
    y_train,
    sample_weight=nb_sample_weights,
)

nb_awff_pred = nb_awff.predict(
    X_test_awff
)

evaluate_model(
    "Naive Bayes",
    "AWFF",
    y_test,
    nb_awff_pred,
)

joblib.dump(
    nb_awff,
    os.path.join(
        MODELS_DIR,
        "naive_bayes_awff.pkl",
    ),
)


# ============================================================
# 19. LOGISTIC REGRESSION — TF-IDF BASELINE
# ============================================================

print(
    "\nTraining Logistic Regression — TF-IDF baseline..."
)

lr_baseline = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    C=1.0,
    random_state=SEED,
)

lr_baseline.fit(
    X_train_tfidf,
    y_train,
)

lr_baseline_pred = lr_baseline.predict(
    X_test_tfidf
)

evaluate_model(
    "Logistic Regression",
    "TF-IDF Baseline",
    y_test,
    lr_baseline_pred,
)


# ============================================================
# 20. LOGISTIC REGRESSION — AWFF
# ============================================================

print(
    "\nTraining Logistic Regression — AWFF..."
)

lr_awff = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    C=1.0,
    random_state=SEED,
)

lr_awff.fit(
    X_train_awff,
    y_train,
)

lr_awff_pred = lr_awff.predict(
    X_test_awff
)

evaluate_model(
    "Logistic Regression",
    "AWFF",
    y_test,
    lr_awff_pred,
)

joblib.dump(
    lr_awff,
    os.path.join(
        MODELS_DIR,
        "logistic_regression_awff.pkl",
    ),
)


# ============================================================
# 21. TOKENIZATION FOR RNN / LSTM
# ============================================================

print("\nPreparing RNN/LSTM inputs...")

tokenizer = Tokenizer(
    num_words=MAX_WORDS,
    oov_token="<OOV>",
)

tokenizer.fit_on_texts(
    train_df["context"]
)

X_train_sequences = tokenizer.texts_to_sequences(
    train_df["context"]
)

X_test_sequences = tokenizer.texts_to_sequences(
    test_df["context"]
)

X_train_pad = pad_sequences(
    X_train_sequences,
    maxlen=MAX_SEQUENCE_LENGTH,
    padding="post",
    truncating="post",
)

X_test_pad = pad_sequences(
    X_test_sequences,
    maxlen=MAX_SEQUENCE_LENGTH,
    padding="post",
    truncating="post",
)

vocab_size = min(
    MAX_WORDS,
    len(tokenizer.word_index) + 1,
)

print("Vocabulary size:", vocab_size)
print(
    "Sequence train shape:",
    X_train_pad.shape,
)
print(
    "Sequence test shape :",
    X_test_pad.shape,
)


# ============================================================
# 22. HELPER — CREATE RECURRENT MODEL
# ============================================================

def build_recurrent_model(model_type):
    """
    Creates either:
      - RNN
      - LSTM

    Inputs:
      1. text sequence
      2. optional AWFF numeric vector

    Baseline:
      numeric vector is all zeros.

    AWFF:
      actual scaled AWFF numeric features.
    """

    text_input = Input(
        shape=(MAX_SEQUENCE_LENGTH,),
        name="text_input",
    )

    awff_input = Input(
        shape=(len(AWFF_NUMERIC_COLUMNS),),
        name="awff_numeric_input",
    )

    embedding = Embedding(
        input_dim=vocab_size,
        output_dim=EMBEDDING_DIM,
        name="embedding",
    )(text_input)

    if model_type == "RNN":
        sequence_output = SimpleRNN(
            RECURRENT_UNITS,
            return_sequences=False,
            name="rnn",
        )(embedding)

    elif model_type == "LSTM":
        sequence_output = LSTM(
            RECURRENT_UNITS,
            return_sequences=False,
            name="lstm",
        )(embedding)

    else:
        raise ValueError(
            "model_type must be 'RNN' or 'LSTM'."
        )

    sequence_output = Dropout(
        0.30
    )(sequence_output)

    combined = Concatenate(
        name="feature_fusion"
    )(
        [
            sequence_output,
            awff_input,
        ]
    )

    combined = Dense(
        96,
        activation="relu",
        name="fusion_dense",
    )(combined)

    combined = Dropout(
        0.30
    )(combined)

    output = Dense(
        num_classes,
        activation="softmax",
        name="sentiment_output",
    )(combined)

    model = Model(
        inputs=[
            text_input,
            awff_input,
        ],
        outputs=output,
        name=f"{model_type}_ABSA",
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# ============================================================
# 23. RNN TRAINING FUNCTION
# ============================================================

def train_recurrent_experiment(
    model_type,
    feature_set,
    numeric_train,
    numeric_test,
):
    print(
        f"\nTraining {model_type} — {feature_set}..."
    )

    model = build_recurrent_model(
        model_type
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
    )

    history = model.fit(
        [
            X_train_pad,
            numeric_train,
        ],
        y_train,
        validation_split=VALIDATION_SPLIT,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=[early_stopping],
        verbose=1,
    )

    probabilities = model.predict(
        [
            X_test_pad,
            numeric_test,
        ],
        verbose=0,
    )

    predictions = np.argmax(
        probabilities,
        axis=1,
    )

    evaluate_model(
        model_type,
        feature_set,
        y_test,
        predictions,
    )

    safe_name = (
        f"{model_type}_{feature_set}"
        .replace(" ", "_")
        .replace("/", "_")
    )

    # Accuracy graph.
    plt.figure(figsize=(8, 5))

    plt.plot(
        history.history["accuracy"],
        label="Training Accuracy",
    )

    plt.plot(
        history.history["val_accuracy"],
        label="Validation Accuracy",
    )

    plt.title(
        f"{model_type} - {feature_set} Accuracy"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            f"{safe_name}_training_accuracy.png",
        ),
        dpi=300,
    )

    plt.close()

    # Loss graph.
    plt.figure(figsize=(8, 5))

    plt.plot(
        history.history["loss"],
        label="Training Loss",
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation Loss",
    )

    plt.title(
        f"{model_type} - {feature_set} Loss"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            f"{safe_name}_training_loss.png",
        ),
        dpi=300,
    )

    plt.close()

    model.save(
        os.path.join(
            MODELS_DIR,
            f"{safe_name.lower()}_model.keras",
        )
    )

    return model


# ============================================================
# 24. ZERO NUMERIC VECTOR FOR TF-IDF RNN/LSTM BASELINE
# ============================================================
#
# The recurrent baseline is text-only. The second input is kept
# as zeros solely so the same model architecture can be used.
# The AWFF experiment receives the actual AWFF numeric features.
# ============================================================

X_train_numeric_zero = np.zeros_like(
    X_train_numeric,
    dtype=np.float32,
)

X_test_numeric_zero = np.zeros_like(
    X_test_numeric,
    dtype=np.float32,
)

X_train_numeric_awff = (
    X_train_numeric.astype(np.float32)
)

X_test_numeric_awff = (
    X_test_numeric.astype(np.float32)
)


# ============================================================
# 25. RNN — TF-IDF / TEXT BASELINE
# ============================================================

rnn_baseline = train_recurrent_experiment(
    "RNN",
    "TF-IDF Baseline",
    X_train_numeric_zero,
    X_test_numeric_zero,
)


# ============================================================
# 26. RNN — AWFF
# ============================================================

rnn_awff = train_recurrent_experiment(
    "RNN",
    "AWFF",
    X_train_numeric_awff,
    X_test_numeric_awff,
)


# ============================================================
# 27. LSTM — TF-IDF / TEXT BASELINE
# ============================================================

lstm_baseline = train_recurrent_experiment(
    "LSTM",
    "TF-IDF Baseline",
    X_train_numeric_zero,
    X_test_numeric_zero,
)


# ============================================================
# 28. LSTM — AWFF
# ============================================================

lstm_awff = train_recurrent_experiment(
    "LSTM",
    "AWFF",
    X_train_numeric_awff,
    X_test_numeric_awff,
)


# ============================================================
# 29. MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(
    all_results
)

results_df = results_df[
    [
        "Model",
        "Feature_Set",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "Macro_F1",
    ]
]

print("\n" + "=" * 90)
print("FINAL MODEL COMPARISON")
print("=" * 90)

print(
    results_df.to_string(
        index=False
    )
)

results_df.to_csv(
    os.path.join(
        RESULTS_DIR,
        "model_comparison_baseline_vs_awff.csv",
    ),
    index=False,
)


# ============================================================
# 30. AWFF IMPROVEMENT TABLE
# ============================================================

improvement_rows = []

for model_name in results_df["Model"].unique():

    baseline_row = results_df[
        (results_df["Model"] == model_name)
        &
        (
            results_df["Feature_Set"]
            == "TF-IDF Baseline"
        )
    ]

    awff_row = results_df[
        (results_df["Model"] == model_name)
        &
        (
            results_df["Feature_Set"]
            == "AWFF"
        )
    ]

    if (
        not baseline_row.empty
        and not awff_row.empty
    ):
        improvement_rows.append(
            {
                "Model": model_name,

                "Accuracy_Baseline":
                    baseline_row.iloc[0]["Accuracy"],

                "Accuracy_AWFF":
                    awff_row.iloc[0]["Accuracy"],

                "Accuracy_Improvement":
                    (
                        awff_row.iloc[0]["Accuracy"]
                        -
                        baseline_row.iloc[0]["Accuracy"]
                    ),

                "F1_Baseline":
                    baseline_row.iloc[0]["F1"],

                "F1_AWFF":
                    awff_row.iloc[0]["F1"],

                "F1_Improvement":
                    (
                        awff_row.iloc[0]["F1"]
                        -
                        baseline_row.iloc[0]["F1"]
                    ),

                "Macro_F1_Baseline":
                    baseline_row.iloc[0]["Macro_F1"],

                "Macro_F1_AWFF":
                    awff_row.iloc[0]["Macro_F1"],

                "Macro_F1_Improvement":
                    (
                        awff_row.iloc[0]["Macro_F1"]
                        -
                        baseline_row.iloc[0]["Macro_F1"]
                    ),
            }
        )

improvement_df = pd.DataFrame(
    improvement_rows
)

print("\n" + "=" * 90)
print("AWFF IMPROVEMENT")
print("=" * 90)

print(
    improvement_df.to_string(
        index=False
    )
)

improvement_df.to_csv(
    os.path.join(
        RESULTS_DIR,
        "awff_improvement_analysis.csv",
    ),
    index=False,
)


# ============================================================
# 31. MODEL ACCURACY COMPARISON
# ============================================================

plt.figure(figsize=(11, 6))

plot_df = results_df.copy()

plot_df["Model_Features"] = (
    plot_df["Model"]
    + " - "
    + plot_df["Feature_Set"]
)

sns.barplot(
    data=plot_df,
    x="Model_Features",
    y="Accuracy",
)

plt.title(
    "Accuracy Comparison: TF-IDF Baseline vs AWFF"
)
plt.xlabel("Model and Feature Set")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.xticks(rotation=25)
plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "accuracy_baseline_vs_awff.png",
    ),
    dpi=300,
)

plt.close()


# ============================================================
# 32. MACRO F1 COMPARISON
# ============================================================

plt.figure(figsize=(11, 6))

sns.barplot(
    data=plot_df,
    x="Model_Features",
    y="Macro_F1",
)

plt.title(
    "Macro F1 Comparison: TF-IDF Baseline vs AWFF"
)
plt.xlabel("Model and Feature Set")
plt.ylabel("Macro F1")
plt.ylim(0, 1)
plt.xticks(rotation=25)
plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "macro_f1_baseline_vs_awff.png",
    ),
    dpi=300,
)

plt.close()


# ============================================================
# 33. ALL METRICS COMPARISON
# ============================================================

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "Macro_F1",
]

for metric in metrics:

    plt.figure(figsize=(11, 6))

    sns.barplot(
        data=plot_df,
        x="Model_Features",
        y=metric,
    )

    plt.title(
        f"{metric} Comparison: TF-IDF Baseline vs AWFF"
    )
    plt.xlabel("Model and Feature Set")
    plt.ylabel(metric)
    plt.ylim(0, 1)
    plt.xticks(rotation=25)
    plt.tight_layout()

    filename = (
        metric.lower()
        + "_baseline_vs_awff.png"
    )

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            filename,
        ),
        dpi=300,
    )

    plt.close()


# ============================================================
# 34. SAVE SUPPORTING OBJECTS
# ============================================================

joblib.dump(
    tfidf,
    os.path.join(
        MODELS_DIR,
        "tfidf_vectorizer.pkl",
    ),
)

joblib.dump(
    scaler,
    os.path.join(
        MODELS_DIR,
        "awff_scaler.pkl",
    ),
)

joblib.dump(
    label_encoder,
    os.path.join(
        MODELS_DIR,
        "label_encoder.pkl",
    ),
)

joblib.dump(
    tokenizer,
    os.path.join(
        MODELS_DIR,
        "tokenizer.pkl",
    ),
)

joblib.dump(
    AWFF_NUMERIC_COLUMNS,
    os.path.join(
        MODELS_DIR,
        "awff_numeric_columns.pkl",
    ),
)


# ============================================================
# 35. SAVE EXPERIMENT CONFIGURATION
# ============================================================

config = {
    "seed": SEED,
    "test_size": TEST_SIZE,
    "tfidf_max_features": TFIDF_MAX_FEATURES,
    "tfidf_ngram_range": TFIDF_NGRAM_RANGE,
    "tfidf_min_df": TFIDF_MIN_DF,
    "tfidf_max_df": TFIDF_MAX_DF,
    "max_words": MAX_WORDS,
    "max_sequence_length": MAX_SEQUENCE_LENGTH,
    "embedding_dim": EMBEDDING_DIM,
    "recurrent_units": RECURRENT_UNITS,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "validation_split": VALIDATION_SPLIT,
    "context_window": CONTEXT_WINDOW,
    "awff_numeric_features": AWFF_NUMERIC_COLUMNS,
    "supervision": (
        "comment-level labels propagated to extracted "
        "aspect records; weak/distant supervision"
    ),
}

pd.Series(config, dtype="object").to_csv(
    os.path.join(
        RESULTS_DIR,
        "experiment_configuration.csv",
    )
)


# ============================================================
# 36. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 90)
print("ABSA MODEL TRAINING COMPLETED")
print("=" * 90)

print("\nResults saved in:")
print(f"  {RESULTS_DIR}/")

print("\nModels saved in:")
print(f"  {MODELS_DIR}/")

print("\nMain result files:")
print("  1. extracted_aspects.csv")
print("  2. model_comparison_baseline_vs_awff.csv")
print("  3. awff_improvement_analysis.csv")
print("  4. experiment_configuration.csv")

print("\nThe experiment compares:")
print("  TF-IDF baseline")
print("  Proposed full AWFF")

print("\nModels:")
print("  Naive Bayes")
print("  Logistic Regression")
print("  RNN")
print("  LSTM")

print("\nIMPORTANT:")
print(
    "Do not claim true human-annotated aspect-level ground truth. "
    "The current dataset provides comment-level sentiment labels."
)
