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

    def linear_SVC(self):
        data = pd.read_csv(self.data_path)

        # Split the data to training / test
        X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

        # Vectorize the text data
        X_train_vectorized = self.vectorizer.fit_transform(X_train)
        X_test_vectorized = self.vectorizer.transform(X_test)

        # Set hyperparameters
        model_params = {
            'C': 0.1,
            'loss': 'hinge',
            'max_iter': 5000,
            'penalty': 'l2',
            'dual': True,
            'tol': 1e-4
        }

        # Initialize and train the model
        model = LinearSVC(**model_params)
        model.fit(X_train_vectorized, y_train)
        self.evaluate_model(model, X_test_vectorized, y_test)
        return model, self.vectorizer

    def logistic_regression(self):
        data = pd.read_csv(self.data_path)

        # Split the data to training / test
        X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

        # Vectorize the text data
        X_train_vectorized = self.vectorizer.fit_transform(X_train)
        X_test_vectorized = self.vectorizer.transform(X_test)

        # Set hyperparameters
        model_params = {
            'C': 0.1,
            'penalty': 'l1',
            'max_iter': 10000,
            'solver': 'saga',
            'random_state': 1
        }

        # Initialize and train the model
        model = LogisticRegression(**model_params)
        model.fit(X_train_vectorized, y_train)
        self.evaluate_model(model, X_test_vectorized, y_test)
        return model, self.vectorizer

    def naive_bayes(self):
        data = pd.read_csv(self.data_path)

        # Split the data to training / test
        X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

        # Vectorize the text data
        X_train_vectorized = self.vectorizer.fit_transform(X_train)
        X_test_vectorized = self.vectorizer.transform(X_test)

        # Set hyperparameters
        model = MultinomialNB(force_alpha=False, alpha=0, fit_prior=False)

        # Initialize and train the model
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