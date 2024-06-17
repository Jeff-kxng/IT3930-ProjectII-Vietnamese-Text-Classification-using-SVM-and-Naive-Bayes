import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample
import joblib
import numpy as np
from typing import List, Tuple

# Configuration
FILE_PATH = 'data.csv'
STOPWORDS_PATH = 'stopwords.csv'
RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_FEATURES = 5000

def preprocess_text(text: str) -> str:
    """
    Preprocesses text by converting to lowercase and removing punctuation, numbers, and special characters.
    """
    text = text.lower()
    text = re.sub(r'[^\w\sÀ-ỹ]', '', text)  # Include Vietnamese character range
    return text

def load_data_from_csv(file_path: str) -> pd.DataFrame:
    """
    Loads and preprocesses the text data along with their labels from a CSV file.
    """
    try:
        data = pd.read_csv(file_path)
        if data.empty:
            raise ValueError("No data found in CSV file!")
        data['text'] = data['text'].apply(preprocess_text)
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: File '{file_path}' not found.")
    except pd.errors.EmptyDataError:
        raise ValueError("Error: Empty CSV file.")
    except Exception as e:
        raise RuntimeError(f"Error loading data from CSV: {e}")

def load_stopwords_from_csv(stopwords_path: str) -> List[str]:
    """
    Loads stopwords from a CSV file.
    """
    try:
        stopwords_df = pd.read_csv(stopwords_path)
        if 'stopword' not in stopwords_df.columns:
            raise ValueError("Column 'stopword' not found in the CSV file.")
        return stopwords_df['stopword'].tolist()
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: File '{stopwords_path}' not found.")
    except ValueError as ve:
        raise ve
    except Exception as e:
        raise RuntimeError(f"Error loading stopwords from CSV: {e}")

def balance_data(X: pd.Series, y: pd.Series, random_state: int = RANDOM_STATE) -> Tuple[pd.Series, pd.Series]:
    """
    Balance the dataset by oversampling the minority class.
    """
    df = pd.concat([X, y], axis=1)
    max_size = df['label'].value_counts().max()

    lst = [df]
    for class_index, group in df.groupby('label'):
        lst.append(group.sample(max_size - len(group), replace=True, random_state=random_state))
    df_new = pd.concat(lst)

    X_balanced = df_new['text']
    y_balanced = df_new['label']

    return X_balanced, y_balanced

def train_and_evaluate(X_train, y_train, X_test, y_test, representation: str) -> None:
    """
    Trains and evaluates an SVM model using the specified representation.
    """
    # Define the model with hyperparameter tuning
    model = SVC(kernel='linear')
    param_grid = {'C': [0.1, 1, 10, 100]}
    stratified_kfold = StratifiedKFold(n_splits=3, random_state=RANDOM_STATE, shuffle=True)  # Ensure reproducibility
    grid = GridSearchCV(model, param_grid, refit=True, verbose=0, cv=stratified_kfold)
    grid.fit(X_train, y_train)

    # Evaluate the model using cross-validation
    cv_scores = cross_val_score(grid.best_estimator_, X_train, y_train, cv=3)
    print(f"Cross-Validation Accuracy for {representation}: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")

    # Plot cross-validation scores
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=np.arange(1, len(cv_scores) + 1), y=cv_scores, marker='o')
    plt.title(f'Cross-Validation Accuracies for {representation}')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.ylim(0.5, 1.05)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Evaluate the model on the test set
    test_score = grid.score(X_test, y_test)
    print(f"Test Accuracy for {representation}: {test_score:.2f}")

    # Print accuracy and classification report
    y_pred = grid.predict(X_test)
    print(classification_report(y_test, y_pred, zero_division=0))

    # Visualize confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=grid.classes_, yticklabels=grid.classes_,
                annot_kws={"fontsize": 12})
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.title(f'Confusion Matrix for {representation}', fontsize=16)
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Save the trained model to a .pkl file
    model_filename = f"{representation.lower().replace(' ', '_')}_svm_model.pkl"
    joblib.dump(grid, model_filename)
    print(f"Trained model saved to {model_filename}")

def main():
    try:
        # Load and preprocess data from CSV
        data = load_data_from_csv(FILE_PATH)
        print(f"Data loaded successfully from '{FILE_PATH}'.")

        # Load Vietnamese stopwords
        vietnamese_stopwords = load_stopwords_from_csv(STOPWORDS_PATH)
        if not vietnamese_stopwords:
            raise ValueError("Error: No stopwords loaded. Check your CSV file.")
        print(f"Stopwords loaded successfully from '{STOPWORDS_PATH}'.")

        X = data['text']
        y = data['label']

        # Balance the dataset
        X, y = balance_data(X, y)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

        # Check the number of samples for each class
        class_counts = y_train.value_counts()
        min_samples = class_counts.min()
        print("Minimum samples per class:", min_samples)

        # Transform text data into TF-IDF representation
        tfidf_vectorizer = TfidfVectorizer(max_features=MAX_FEATURES, stop_words=vietnamese_stopwords)
        X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
        X_test_tfidf = tfidf_vectorizer.transform(X_test)

        # Train and evaluate the model using TF-IDF
        train_and_evaluate(X_train_tfidf, y_train, X_test_tfidf, y_test, "TF-IDF")

        # Transform text data into Bag of Words representation
        bow_vectorizer = CountVectorizer(max_features=MAX_FEATURES, stop_words=vietnamese_stopwords)
        X_train_bow = bow_vectorizer.fit_transform(X_train)
        X_test_bow = bow_vectorizer.transform(X_test)

        # Train and evaluate the model using Bag of Words
        train_and_evaluate(X_train_bow, y_train, X_test_bow, y_test, "Bag of Words")
    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()
