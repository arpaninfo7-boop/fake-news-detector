import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load dataset
data = pd.read_csv("news.csv")

# Features & Labels
X = data['text']
y = data['label']

# Convert text to numerical form
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_vectorized = vectorizer.fit_transform(X)

# Train model
model = LogisticRegression()
model.fit(X_vectorized, y)

# Prediction function
def predict_news(news_text):
    transformed = vectorizer.transform([news_text])
    prediction = model.predict(transformed)
    return prediction[0]

# Confidence score function
def predict_confidence(news_text):
    transformed = vectorizer.transform([news_text])
    prob = model.predict_proba(transformed)
    return prob.max()