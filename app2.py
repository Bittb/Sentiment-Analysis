import pandas as pd
import numpy as np
import re
import joblib
import os
from flask import Flask, request, render_template, send_file
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

app = Flask(__name__)

stop_words = set([...])  # Keep your stop words as-is

def preprocess_text(text):
    if pd.isnull(text):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = re.findall(r'\b\w{3,}\b', text)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Load or train model
if os.path.exists('sentiment_model.pkl') and os.path.exists('tfidf_vectorizer.pkl'):
    model = joblib.load('sentiment_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
else:
    df = pd.read_csv('product_reviews.csv')
    df['Review'] = df['Review'].astype(str)
    df['clean_review'] = df['Review'].apply(preprocess_text)
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['clean_review'])
    y = df['Sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # ✅ Use multinomial (softmax) logistic regression
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
    model.fit(X_train, y_train)
    joblib.dump(model, 'sentiment_model.pkl')
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

def predict_sentiment(text):
    text = preprocess_text(text)
    vector = vectorizer.transform([text])
    prediction = model.predict(vector)[0]
    return prediction

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        review = request.form['review']
        prediction = predict_sentiment(review)
        return render_template('index.html', review=review, prediction=prediction)

@app.route('/bulk_predict', methods=['POST'])
def bulk_predict():
    try:
        file = request.files['file']
        if not file:
            return render_template('bulk_result.html', error="No file uploaded.")

        df = pd.read_csv(file)
        if 'Review' not in df.columns:
            return render_template('bulk_result.html', error="CSV must contain a 'Review' column.")

        df['Review'] = df['Review'].astype(str)
        df['clean_review'] = df['Review'].apply(preprocess_text)
        vectors = vectorizer.transform(df['clean_review'])
        df['Predicted_Sentiment'] = model.predict(vectors)

        pos = (df['Predicted_Sentiment'] == 'positive').sum()
        neg = (df['Predicted_Sentiment'] == 'negative').sum()
        neu = (df['Predicted_Sentiment'] == 'neutral').sum()

        chart_path = 'static/piechart.png'
        if os.path.exists(chart_path):
            os.remove(chart_path)

        labels = ['Positive', 'Negative', 'Neutral']
        sizes = [pos, neg, neu]
        colors = ['#28a745', '#dc3545', '#ffc107']

        plt.switch_backend('agg')
        plt.figure(figsize=(6,6))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
        plt.title('Sentiment Distribution')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(chart_path)
        plt.close()

        df.to_csv('static/analyzed_reviews.csv', index=False)

        return render_template('bulk_result.html', pos=pos, neg=neg, neu=neu,
                               piechart=True, tables=[df.to_html(classes='data')],
                               titles=df.columns.values)

    except Exception as e:
        print(f"[ERROR] Bulk predict failed: {e}")
        return render_template('bulk_result.html', error=f"An unexpected error occurred: {str(e)}")


if __name__ == "__main__":
    app.run(debug=True)

