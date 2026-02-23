# =====================================================
# 1. Import Libraries
# =====================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from wordcloud import WordCloud

nltk.download('stopwords')

# =====================================================
# 2. Load Dataset
# =====================================================
data = pd.read_csv("D:\Oasis internship\project 4 of level 1\cleaned_sentiment_dataset.csv")

print("Dataset Shape:", data.shape)
print(data.head())

# Keep only required columns
data = data[['Translated_Review', 'Sentiment']]
data = data.dropna()

# =====================================================
# 3. Text Preprocessing (NLP)
# =====================================================
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

data['Cleaned_Text'] = data['Translated_Review'].apply(clean_text)

# =====================================================
# 4. Feature Engineering (TF-IDF)
# =====================================================
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['Cleaned_Text'])
y = data['Sentiment']

# =====================================================
# 5. Train-Test Split
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================================================
# 6. Machine Learning Models
# =====================================================

# ----- Naive Bayes -----
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
nb_pred = nb_model.predict(X_test)

print("\nNaive Bayes Accuracy:", accuracy_score(y_test, nb_pred))
print("\nNaive Bayes Report:\n", classification_report(y_test, nb_pred))


# ----- Support Vector Machine -----
svm_model = LinearSVC()
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)

print("\nSVM Accuracy:", accuracy_score(y_test, svm_pred))
print("\nSVM Report:\n", classification_report(y_test, svm_pred))

# =====================================================
# 7. Confusion Matrix Visualization
# =====================================================
plt.figure()
sns.heatmap(confusion_matrix(y_test, svm_pred), 
            annot=True, fmt='d')
plt.title("Confusion Matrix (SVM)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# =====================================================
# 8. Sentiment Distribution Visualization
# =====================================================
plt.figure()
data['Sentiment'].value_counts().plot(kind='bar')
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()

# =====================================================
# 9. Word Cloud Visualization
# =====================================================
all_words = " ".join(data['Cleaned_Text'])
wordcloud = WordCloud(width=800, height=400).generate(all_words)

plt.figure()
plt.imshow(wordcloud)
plt.axis("off")
plt.title("Word Cloud")
plt.show()
