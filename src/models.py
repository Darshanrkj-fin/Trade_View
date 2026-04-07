import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

def create_vectorizer(max_df=0.8, ngram_range=(1, 2)):
    """
    Creates and returns a TF-IDF vectorizer.
    """
    return TfidfVectorizer(max_df=max_df, ngram_range=ngram_range)

def train_ensemble_model(tfidf_train, y_train):
    """
    Trains an ensemble of SVM, Naive Bayes, Random Forest, and AdaBoost.
    Returns the fitted model.
    """
    svm_clf = SVC(kernel='linear', probability=True)
    nb_clf = MultinomialNB()
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    ada_clf = AdaBoostClassifier(n_estimators=50, random_state=42)

    ensemble_model = VotingClassifier(
        estimators=[
            ('svm', svm_clf),
            ('nb', nb_clf),
            ('rf', rf_clf),
            ('ada', ada_clf)
        ],
        voting='soft'
    )

    ensemble_model.fit(tfidf_train, y_train)
    return ensemble_model

def get_performance_report(model, tfidf_test, y_test):
    """
    Calculates and returns accuracy and classification report.
    """
    y_pred = model.predict(tfidf_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report

def save_model_and_vectorizer(model, vectorizer, model_path='ensemble_model.pkl', vectorizer_path='tfidf_vectorizer.pkl'):
    """
    Serializes and saves the model and vectorizer to disk.
    """
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

def load_model_and_vectorizer(model_path='ensemble_model.pkl', vectorizer_path='tfidf_vectorizer.pkl'):
    """
    Loads and returns the serialized model and vectorizer.
    """
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer
