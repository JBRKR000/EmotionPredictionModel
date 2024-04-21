import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
import tkinter as tk


class Model:
    def __init__(self, data_path="emotions-dataset.csv"):
        self.data_path = data_path
        self.vectorizer = TfidfVectorizer(max_features=5000)

    def linear_SVC(self, C=0.1, loss='squared_hinge', max_iter=2000, penalty='l2', dual=False, tol=1e-2):
        data = pd.read_csv(self.data_path)
        X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)
        X_train_vectorized = self.vectorizer.fit_transform(X_train)
        X_test_vectorized = self.vectorizer.transform(X_test)

        model = LinearSVC(C=C, loss=loss, max_iter=max_iter, penalty=penalty, dual=dual, tol=tol)
        model.fit(X_train_vectorized, y_train)
        self.evaluate_model(model, X_test_vectorized, y_test)
        return model, self.vectorizer

    def logistic_regression(self, C=0.001, penalty=None, max_iter=5000, random_state=1):
        data = pd.read_csv(self.data_path)
        X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)
        X_train_vectorized = self.vectorizer.fit_transform(X_train)
        X_test_vectorized = self.vectorizer.transform(X_test)

        model = LogisticRegression(C=C, penalty=penalty, max_iter=max_iter, random_state=random_state)
        model.fit(X_train_vectorized, y_train)
        self.evaluate_model(model, X_test_vectorized, y_test)
        return model, self.vectorizer

    def naive_bayes(self, alpha=0.01, fit_prior=True, class_prior=None):
        data = pd.read_csv(self.data_path)
        X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)
        X_train_vectorized = self.vectorizer.fit_transform(X_train)
        X_test_vectorized = self.vectorizer.transform(X_test)

        model = MultinomialNB(alpha=alpha, fit_prior=fit_prior, class_prior=class_prior)
        model.fit(X_train_vectorized, y_train)
        self.evaluate_model(model, X_test_vectorized, y_test)
        return model, self.vectorizer

    def evaluate_model(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy:", accuracy)
        report = classification_report(y_test, y_pred)
        print("Classification Report:")
        print(report)
        self.display_report(report)

    @staticmethod
    def display_report(report):
        root = tk.Tk()
        root.title("Classification Report")
        text_area = tk.Text(root, height=20, width=80)
        text_area.insert(tk.END, report)
        text_area.pack()
        root.mainloop()

    def predict_emotion(self, input_text, model):
        input_text_vectorized = self.vectorizer.transform([input_text])
        predicted_emotion = model.predict(input_text_vectorized)
        return predicted_emotion[0]