# 📰 Fake News Detection App

> 🚀 **An NLP and Machine Learning-based web application for detecting whether a news article is REAL or FAKE using TF-IDF feature extraction and multiple classification models.**

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![NLTK](https://img.shields.io/badge/NLTK-NLP-green)](https://www.nltk.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red?logo=streamlit)](https://streamlit.io/)

---

## 📌 Project Overview

The **Fake News Detection App** is a Natural Language Processing (NLP) and Machine Learning project designed to automatically classify news articles as **REAL** or **FAKE**.

The system uses **TF-IDF (Term Frequency–Inverse Document Frequency)** to transform textual news articles into numerical features and evaluates multiple machine learning algorithms, including:

* Logistic Regression
* LinearSVC
* Naive Bayes
* Random Forest

A **Streamlit-based interactive web application** allows users to enter a news article and receive an immediate prediction together with a confidence score.

---

## 🎯 Project Objectives

The main objectives of this project are to:

* Develop an automated fake news classification system.
* Apply NLP techniques to unstructured news text.
* Convert textual data into numerical features using TF-IDF.
* Compare multiple machine learning algorithms.
* Evaluate models using standard classification metrics.
* Provide an interactive web interface for real-time predictions.
* Analyze differences between REAL and FAKE news using visualizations.

---

## ✨ Key Features

### 🧹 Text Cleaning & Preprocessing

The input text is processed before model prediction.

* Convert text to lowercase
* Remove punctuation and special characters
* Remove stopwords using NLTK
* Tokenization
* Text normalization

---

### 🤖 Machine Learning Models

The project evaluates several classification algorithms:

| Model                   | Purpose                                        |
| ----------------------- | ---------------------------------------------- |
| **Logistic Regression** | Strong linear classification baseline          |
| **LinearSVC**           | Support Vector Machine for text classification |
| **Naive Bayes**         | Efficient probabilistic text classifier        |
| **Random Forest**       | Ensemble-based nonlinear classifier            |

This model comparison helps identify the most suitable algorithm for the fake news detection task.

---

### 📈 Prediction Confidence

The application provides a confidence/certainty indication alongside the prediction.

Example:

```text
Prediction: REAL
Confidence: 94.52%
```

The confidence estimation helps users understand how strongly the model supports its classification.

> **Note:** A machine-learning confidence score should not be interpreted as a guarantee that an article is factually true or false.

---

### 🌐 Interactive Streamlit Web App

Users can:

1. Enter or paste a news article.
2. Submit the article for analysis.
3. Receive an instant REAL/FAKE prediction.
4. View the associated confidence score.

---

## 📊 Data Visualization

The project includes several visual analyses to understand the dataset and model performance.

### ☁️ Word Clouds

Word clouds can be generated separately for:

* REAL news
* FAKE news

This provides a visual representation of frequently occurring terms in each category.

### 📉 Model Evaluation Visualizations

The project also includes:

* Confusion Matrix
* ROC Curve
* AUC Score
* News Length Distribution

These visualizations help analyze both the dataset and classification performance.

---

## 🗂 Dataset

The project uses two primary datasets:

| Dataset    | Description        | Label |
| ---------- | ------------------ | ----: |
| `Fake.csv` | Fake news articles |   `0` |
| `True.csv` | Real news articles |   `1` |

### Label Mapping

```text
0 → FAKE
1 → REAL
```

The datasets contain textual news articles that are transformed into machine-readable features during preprocessing.

---

# 🧠 System Architecture

```text
                 ┌─────────────────────┐
                 │     News Article    │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Text Preprocessing  │
                 │ Lowercase           │
                 │ Stopword Removal    │
                 │ Cleaning            │
                 │ Tokenization        │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │  TF-IDF Vectorizer  │
                 └──────────┬──────────┘
                            │
                            ▼
              ┌─────────────────────────────┐
              │     ML Classification       │
              ├─────────────────────────────┤
              │ Logistic Regression         │
              │ LinearSVC                   │
              │ Naive Bayes                 │
              │ Random Forest               │
              └─────────────┬───────────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ REAL / FAKE         │
                 │ Prediction          │
                 │ Confidence Score    │
                 └─────────────────────┘
```

---

# 🔄 Model Training Workflow

```text
Dataset
   ↓
Data Loading
   ↓
Text Cleaning
   ↓
Train / Test Split
   ↓
TF-IDF Feature Extraction
   ↓
Model Training
   ↓
Cross-Validation
   ↓
GridSearchCV
   ↓
Model Evaluation
   ↓
Best Model Selection
   ↓
Save Model + Vectorizer
   ↓
Streamlit Deployment
```

---

## 🧪 Model Evaluation & Tuning

The models are evaluated using standard machine learning metrics:

### Accuracy

Measures the overall percentage of correctly classified news articles.

### Precision

Measures how many articles predicted as fake are actually fake.

### Recall

Measures how many actual fake articles are correctly detected.

### F1-Score

Provides a balance between precision and recall.

### ROC-AUC

Measures the model's ability to distinguish between REAL and FAKE news across classification thresholds.

### Confusion Matrix

Provides a detailed breakdown of:

* True Positives
* True Negatives
* False Positives
* False Negatives

---

## ⚙️ Hyperparameter Optimization

To improve model performance, the project uses:

* Cross-validation
* `GridSearchCV`
* Model comparison
* Classification reports
* ROC-AUC analysis

The best-performing model can then be selected for deployment.

---

# 💾 Model Persistence

After training, the selected model and TF-IDF vectorizer are saved using Python's `pickle` module.

```text
saved_model.pkl
tfidf_vectorizer.pkl
```

This allows the Streamlit application to load the trained components directly without retraining the model every time the application starts.

---

# 🛠️ Tech Stack

### Programming Language

* Python 3.x

### Machine Learning & NLP

* Pandas
* NumPy
* Scikit-learn
* NLTK

### Visualization

* Matplotlib
* Seaborn
* WordCloud

### Web Application

* Streamlit

### Model Persistence

* Pickle

---

# 📦 Project Structure

```text
fake-news-detector/
│
├── app.py
├── Fake.csv
├── True.csv
├── saved_model.pkl
├── tfidf_vectorizer.pkl
├── Fake_News.ipynb
├── requirements.txt
├── LLM.JPG
└── README.md
```

### File Description

| File                   | Description                                          |
| ---------------------- | ---------------------------------------------------- |
| `app.py`               | Streamlit application                                |
| `Fake.csv`             | Fake news dataset                                    |
| `True.csv`             | Real news dataset                                    |
| `saved_model.pkl`      | Trained machine learning model                       |
| `tfidf_vectorizer.pkl` | Saved TF-IDF vectorizer                              |
| `Fake_News.ipynb`      | Data preprocessing, training and evaluation notebook |
| `requirements.txt`     | Project dependencies                                 |
| `LLM.JPG`              | Application screenshot                               |

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/yourusername/fake-news-detector.git
cd fake-news-detector
```

> Replace the repository URL above with your actual GitHub repository URL.

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Download NLTK Stopwords

Run Python:

```python
import nltk
nltk.download('stopwords')
```

---

# 🚀 Usage

Start the Streamlit application:

```bash
streamlit run app.py
```

Then open the application in your browser:

```text
http://localhost:8501
```

Enter a news article and submit it for classification.

The application will return:

```text
Prediction → REAL / FAKE
Confidence → XX.XX%
```

---

# 🧠 Model Training Process

The complete training pipeline follows these steps:

### Step 1 — Data Loading

Load the REAL and FAKE news datasets.

### Step 2 — Text Preprocessing

The `clean_text()` function performs operations such as:

```text
Lowercasing
     ↓
Remove punctuation
     ↓
Remove special characters
     ↓
Remove stopwords
     ↓
Tokenization
```

### Step 3 — TF-IDF Feature Extraction

The cleaned text is converted into numerical vectors using:

```python
TfidfVectorizer()
```

### Step 4 — Model Training

Multiple machine learning models are trained:

```text
Logistic Regression
Naive Bayes
Random Forest
LinearSVC
```

### Step 5 — Evaluation

Models are evaluated using:

```text
Accuracy
Precision
Recall
F1-Score
Confusion Matrix
ROC-AUC
```

### Step 6 — Model Selection

The best-performing model is selected based on the evaluation results.

### Step 7 — Model Persistence

The trained model and TF-IDF vectorizer are saved using:

```text
pickle
```

---

# 🌍 Applications

This project can be used as a foundation for:

* 📰 News credibility analysis
* 🔎 Automated content screening
* 📱 Social media misinformation detection
* 🧠 NLP research
* 📊 Media analytics
* 🚨 Early-stage misinformation monitoring
* 🎓 Educational demonstrations of NLP classification

---

# 🚀 Future Improvements

The project can be extended in several directions:

### 🤖 Deep Learning

* LSTM
* GRU
* CNN for text classification
* Transformer-based models
* BERT
* RoBERTa

### 🧠 Advanced NLP

* Word embeddings
* Word2Vec
* GloVe
* FastText
* Transformer embeddings
* Semantic similarity analysis

### 🌐 Deployment

* Deploy using Streamlit Cloud
* AWS deployment
* Render deployment
* Docker containerization
* REST API integration

### 📊 Explainable AI

Future versions could explain **why** an article was classified as fake or real using:

* Feature importance
* SHAP
* LIME
* Important-word highlighting

---

# 🔬 Research Perspective

This project demonstrates a complete NLP classification pipeline:

```text
Raw Text
   ↓
Text Preprocessing
   ↓
Feature Engineering
   ↓
TF-IDF Representation
   ↓
Machine Learning
   ↓
Model Evaluation
   ↓
Deployment
```

It provides a practical foundation for further research into **misinformation detection, NLP classification, explainable AI, and transformer-based fact verification systems**.

---

# 🤝 Contribution

Contributions and improvements are welcome.

You can:

1. Fork the repository.
2. Create a new branch.
3. Implement your changes.
4. Commit your changes.
5. Open a Pull Request.

---

# 📄 License

This project is intended for **educational and research purposes**.


⭐ **If you find this project useful, consider giving the repository a star!**
