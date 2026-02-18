📰 Fake News Detection App
Detect whether a news article is REAL or FAKE using Machine Learning! 🚀 This app uses TF-IDF and Logistic Regression / LinearSVC to classify news with a confidence score.
✨ Features

🧹 Text Cleaning & Preprocessing
Removes stopwords, special characters, and converts text to lowercase.

🤖 Machine Learning Models

Logistic Regression

LinearSVC

Naive Bayes

Random Forest

📈 Prediction Confidence
Shows % certainty of the prediction using a sigmoid-based score.

🌐 Interactive Web App
Enter any news article and get instant predictions.

📊 Visualizations

Word Clouds for FAKE and REAL news

Confusion Matrix & ROC Curve

News Length Distribution

⚙️ Model Evaluation & Tuning
Cross-validation and GridSearchCV for optimal performance.

📦 Installation:
git clone https://github.com/yourusername/fake-news-detector.git
cd fake-news-detector

Install dependencies:
pip install -r requirements.txt

Download NLTK stopwords:
import nltk
nltk.download('stopwords')

🛠 Usage
Run the app: streamlit run app.py

🗂 Dataset
Fake.csv – Fake news articles (label 0)
True.csv – Real news articles (label 1)

📊 Visualization
Word Clouds for top words in FAKE vs REAL news
Confusion Matrix heatmap
ROC curve & AUC score

⚙️ Model Training
Preprocess text with clean_text()
Convert text using TF-IDF Vectorizer
Train models (Logistic Regression, Naive Bayes, Random Forest, SVM)
Evaluate using accuracy, classification report, confusion matrix, ROC-AUC
Save best model & vectorizer with pickle for deployment

🖥 Tech Stack
Python 3.x
🐍 Libraries: pandas, numpy, scikit-learn, nltk, matplotlib, wordcloud, streamlit