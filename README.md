# 📈 Advanced NLP Sentiment Analysis Engine

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Framework-FF4B4B.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-F7931E.svg)
![NLTK](https://img.shields.io/badge/NLTK-NLP-154f5b.svg)

## 📌 Overview
An end-to-end, highly robust Sentiment Analysis Data Pipeline designed specifically to tackle the complexities of unstructured social media text (Twitter data). 

Traditional Machine Learning models often fail on social media data due to **Negation Mishandling** (where standard stopwords remove crucial words like "not" or "didn't") and the **Out-of-Vocabulary (OOV)** problem. This project solves these issues by implementing a **Hybrid AI Architecture** that combines the statistical pattern recognition of **Logistic Regression** (using TF-IDF) with the grammatical intelligence of a rule-based lexicon (**VADER**).

## ✨ Key Features
* **Custom Token Concatenation:** A targeted Regex pipeline that binds negation operators to their targets (e.g., automatically converting `"not good"` to `"not_good"`) before stopword removal, completely preserving the sentence's semantic polarity.
* **Class Imbalance Handling:** Strategically pruned the ambiguous 'Neutral' class to reframe the system as a strict binary classifier, eliminating Dominant Feature Bias.
* **Hybrid Inference Logic:** The system dynamically overrides the ML predictions using VADER polarity scores when explicit negation tokens are detected, ensuring high contextual accuracy even on unseen slang.
* **Dual-Mode Web Application:** * **Single Text Analysis:** For real-time sentiment testing and debugging.
    * **Bulk CSV Dashboard:** Capable of processing thousands of rows concurrently, generating an automated data dashboard with Matplotlib pie charts and a downloadable results file.

## 🛠️ Technology Stack
* **Language:** Python 3.10+
* **Frontend Framework:** Streamlit
* **Machine Learning:** Scikit-Learn (Logistic Regression, TF-IDF Vectorizer)
* **Natural Language Processing:** NLTK (VADER Lexicon, Stopwords)
* **Data Manipulation & Viz:** Pandas, NumPy, Matplotlib, Regular Expressions (Regex)

## 🚀 How to Run the Project Locally

### Prerequisites
Make sure you have Python installed on your system.

### Step 1: Clone the Repository
```bash
git clone [https://github.com/abhishekdaredavid/Sentiment-Analysis-Hybrid-NLP.git](https://github.com/YOUR_USERNAME/Sentiment-Analysis-Hybrid-NLP.git)
cd Sentiment-Analysis-Hybrid-NLP
Step 2: Install Dependencies
Install all the required Python libraries using the requirements.txt file:

Bash
pip install -r requirements.txt
Step 3: Run the Streamlit App
Launch the web interface by executing the following command in your terminal:

Bash
python -m streamlit run app.py
The application will automatically open in your default web browser at http://localhost:8501.

📂 Project Structure
Plaintext
Sentiment-Analysis-Hybrid-NLP/
│
├── 1_data_exploration.ipynb   # Backend Jupyter Notebook (Data Cleaning & Model Training)
├── app.py                     # Frontend Streamlit Application UI
├── sentiment_model.pkl        # Serialized Logistic Regression Model
├── tfidf_vectorizer.pkl       # Serialized TF-IDF Feature Extractor
├── twitter_dataset.csv        # Raw dataset utilized for training
├── requirements.txt           # List of dependencies
└── README.md                  # Project Documentation
🧠 The Hybrid Logic Explained
Text Input: User inputs a sentence (e.g., "This phone is not good").

Custom Normalization: The system applies token concatenation -> "This phone is not_good".

ML Prediction: The TF-IDF + Logistic Regression model makes an initial statistical prediction.

VADER Rule-Check: The system simultaneously calculates the VADER compound score. If strong negation is detected grammatically, VADER overrides the ML model to prevent false positives.

Output: Highly accurate contextual sentiment is displayed.
