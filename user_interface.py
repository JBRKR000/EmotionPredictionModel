from PyQt5.QtCore import pyqtSignal, QThread
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTextEdit, QPushButton, QLabel, QComboBox, QMessageBox, QProgressBar
from model import Model

emotion_mapping = {
    0: 'Sadness',
    1: 'Joy',
    2: 'Love',
    3: 'Anger',
    4: 'Fear',
    5: 'Surprise'
}

EmotionClassifier = Model()
chosen_model = None


class ModelLearningThread(QThread):
    progress_changed = pyqtSignal(int)

    def __init__(self, window, model_func):
        super().__init__()
        self.window = window
        self.model_func = model_func

    def run(self):
        try:
            self.progress_changed.emit(50)
            self.window.model, self.window.vectorizer = self.model_func()
            self.progress_changed.emit(100)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")


class UserInterface(QWidget):
    learning_finished = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.model = EmotionClassifier
        self.vectorizer = None
        self.initUI()
        self.learning_finished.connect(self.display_learning_finished)

    def initUI(self):
        self.setWindowTitle('Emotions Prediction App')
        self.setGeometry(100, 100, 400, 300)
        layout = QVBoxLayout()

        with open('styles.css', 'r') as file:
            style = file.read()
            self.setStyleSheet(style)

        self.model_label = QLabel("Choose model:")
        layout.addWidget(self.model_label)

        self.model_combobox = QComboBox()
        self.model_combobox.addItem("linear_SVC")
        self.model_combobox.addItem("logistic_regression")
        self.model_combobox.addItem("naive_bayes")
        layout.addWidget(self.model_combobox)

        self.learn_button = QPushButton('Learn')
        self.learn_button.clicked.connect(self.learn_model)
        layout.addWidget(self.learn_button)

        self.text_edit = QTextEdit()
        layout.addWidget(self.text_edit)

        self.predict_button = QPushButton('Predict')
        self.predict_button.clicked.connect(self.predict)
        layout.addWidget(self.predict_button)

        self.emotion_display = QTextEdit()
        self.emotion_display.setReadOnly(True)
        layout.addWidget(self.emotion_display)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)

        self.setLayout(layout)

        self.predict_button.setEnabled(False)
        self.text_edit.setEnabled(False)

    def learn_model(self):
        try:
            global chosen_model
            selected_model = self.model_combobox.currentText()

            if selected_model == "linear_SVC":
                chosen_model = EmotionClassifier.linear_SVC
                model_func = EmotionClassifier.linear_SVC
            elif selected_model == "logistic_regression":
                chosen_model = EmotionClassifier.logistic_regression
                model_func = EmotionClassifier.logistic_regression
            elif selected_model == "naive_bayes":
                chosen_model = EmotionClassifier.naive_bayes
                model_func = EmotionClassifier.naive_bayes
            else:
                return

            # Disable buttons and text box while learning
            self.predict_button.setEnabled(False)
            self.text_edit.setEnabled(False)
            self.learn_button.setEnabled(False)
            self.model_combobox.setEnabled(False)

            # Reset progress bar to 0
            self.progress_bar.setValue(0)

            # Start the learning thread
            self.learning_thread = ModelLearningThread(self, model_func)
            self.learning_thread.progress_changed.connect(self.update_progress)
            self.learning_thread.finished.connect(self.finish_learning)
            self.learning_thread.start()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")

    # Update learning progress bar
    def update_progress(self, value):
        self.progress_bar.setValue(value)

    # Emit finish signal
    def finish_learning(self):
        self.learning_finished.emit()

    # Display information about successful learn process
    def display_learning_finished(self):
        QMessageBox.information(self, "Model Learning", "Model learning finished successfully.")
        self.learn_button.setEnabled(True)
        self.model_combobox.setEnabled(True)
        self.predict_button.setEnabled(True)
        self.text_edit.setEnabled(True)

    # Predict user input with chosen model
    def predict(self):
        if self.model is None:
            return

        input_text = self.text_edit.toPlainText()
        print("Predicting from input text:", input_text)
        predicted_emotion = EmotionClassifier.predict_emotion(input_text, self.model)
        mapped_emotion = emotion_mapping[predicted_emotion]
        self.emotion_display.setText(f"Emotion: {mapped_emotion}")
