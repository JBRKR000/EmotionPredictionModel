import tkinter as tk

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


def linear_SVC():
    data = pd.read_csv("emotions-dataset.csv")
    X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer(max_features=1000)
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)
    C = 0.1
    loss = 'squared_hinge'
    max_iter = 2000
    penalty = 'l2'
    dual = False
    tol = 1e-2
    model = LinearSVC(C=C, loss=loss, max_iter=max_iter, penalty=penalty, dual=dual, tol=tol)
    model.fit(X_train_vectorized, y_train)
    # Ocena modelu na zbiorze testowym
    y_pred = model.predict(X_test_vectorized)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    # Raport klasyfikacji
    a = classification_report(y_test, y_pred)
    print("Classification Report:")
    print(a)
    # Tworzenie okna Tkinter do wyświetlenia raportu
    root = tk.Tk()
    root.title("Classification Report")
    text_area = tk.Text(root, height=20, width=80)
    text_area.insert(tk.END, a)
    text_area.pack()
    root.mainloop()
    return model, vectorizer


def logistic_regression_classifier(C=0.001, penalty=None, max_iter=5000, random_state=1):
    data = pd.read_csv("emotions-dataset.csv")
    X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer(max_features=1000)
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)
    model = LogisticRegression(C=C, penalty=penalty, max_iter=max_iter, random_state=random_state)
    model.fit(X_train_vectorized, y_train)
    # Ocena modelu na zbiorze testowym
    y_pred = model.predict(X_test_vectorized)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    # Raport klasyfikacji
    a = classification_report(y_test, y_pred)
    print("Classification Report:")
    print(a)
    # Tworzenie okna Tkinter do wyświetlenia raportu
    root = tk.Tk()
    root.title("Classification Report")
    text_area = tk.Text(root, height=20, width=80)
    text_area.insert(tk.END, a)
    text_area.pack()
    root.mainloop()
    return model, vectorizer


def naive_bayes(alpha=0.01, fit_prior=True, class_prior=None):
    data = pd.read_csv("emotions-dataset.csv")
    X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer(max_features=1000)
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)
    model = MultinomialNB(alpha=alpha, fit_prior=fit_prior, class_prior=class_prior)
    model.fit(X_train_vectorized, y_train)
    # Ocena modelu na zbiorze testowym
    y_pred = model.predict(X_test_vectorized)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    # Raport klasyfikacji
    a = classification_report(y_test, y_pred)
    print("Classification Report:")
    print(a)
    # Tworzenie okna Tkinter do wyświetlenia raportu
    root = tk.Tk()
    root.title("Classification Report")
    text_area = tk.Text(root, height=20, width=80)
    text_area.insert(tk.END, a)
    text_area.pack()
    root.mainloop()
    return model, vectorizer

def predict_emotion(input_text, model, vectorizer):
    example_texts_vectorized = vectorizer.transform([input_text])
    predicted_emotion = model.predict(example_texts_vectorized)
    return predicted_emotion[0]

