import pandas as pd
import numpy as np
import nltk
import string
import re

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

df = pd.read_csv("dataset.csv", sep='\t', header=None, names=['label', 'text'], encoding='latin-1')
df = df[df['label'].isin(['ham', 'spam'])].dropna(subset=['text']).reset_index(drop=True)

print("\nOriginal Data:\n", df.head())

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    tokens = word_tokenize(text)
    processed = [
        lemmatizer.lemmatize(stemmer.stem(word))
        for word in tokens if word not in stop_words
    ]
    return " ".join(processed)

df['processed_text'] = df['text'].apply(preprocess)

print("\nProcessed Data:\n", df[['text', 'processed_text']].head())

tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['processed_text']).toarray()

y = df['label'].map({'ham': 0, 'spam': 1})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

while True:
    user_input = input("\nEnter message (or 'exit'): ")
    
    if user_input.lower() == "exit":
        break
    
    processed_input = preprocess(user_input)
    vector_input = tfidf.transform([processed_input]).toarray()
    
    result = model.predict(vector_input)
    
    if result[0] == 1:
        print("Spam Message")
    else:
        print("Ham Message")