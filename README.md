# Emotions Prediction Model

This project is a university assignment aimed at predicting emotions from text data using machine learning techniques.

## Dataset
The dataset used in this project was obtained from [Kaggle](https://www.kaggle.com/datasets/nelgiriyewithana/emotions). It contains a collection of English text samples labeled with different emotions.

### Emotion Labels
Each entry in this dataset consists of a text segment representing a Twitter message and a corresponding label indicating the predominant emotion conveyed. The emotions are classified into six categories:
- Sadness (0) ☹️
- Joy (1) 😄
- Love (2) 🥰
- Anger (3) 😠
- Fear (4) 😨
- Surprise (5) 😲

## Description
The Emotions Prediction Model project focuses on developing a machine learning model to classify emotions expressed in text data. The model is trained on the provided dataset and then used to predict the emotions associated with new text inputs.

## Technologies Used
- Python
- Scikit-learn
- Pandas
- PyQt5 (for the GUI)

## How to Use
 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/gist/wiktorlewandowski9/3dbe22591f8ff9b270360376f86bd92b/emotions-prediction-model.ipynb)
[![Open In Gist](https://img.shields.io/badge/Open%20in%20Gist-black?logo=github)](https://gist.github.com/wiktorlewandowski9/3dbe22591f8ff9b270360376f86bd92b)

Or check out simple GUI version:
1. **Clone the repository to your local machine.**
    ```bash
    git clone https://github.com/JBRKR000/EmotionPredictionModel.git
    ```

2. **Install the required dependencies listed in the [`requirements.txt`](link_do_requirements.txt) file.**
    ```bash
    pip install -r requirements.txt
    ```

3. **Download and insert the `emotions-dataset.csv` file into your main program folder.**

4. **Run the `main.py` file to launch the GUI application.**
    ```bash
    python main.py
    ```
    
5. **Enter the text input in the provided text field.**

6. **Click on the "Predict" button to see the predicted emotion.**

## Contributors
- Wiktor Lewandowski
- Jakub Kozimor

## License
MIT License

