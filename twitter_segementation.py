# ===============================
# 1. Install Required Libraries (Run once if needed)
# ===============================
# pip install pandas numpy matplotlib seaborn scikit-learn nltk wordcloud

# ===============================
# 2. Import Libraries
# ===============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from wordcloud import WordCloud

nltk.download('stopwords')

# ===============================
# 3. Load Dataset
# ===============================
data = pd.read_csv("D:\Oasis internship\project 4 of level 1\Twitter_Data.csv")
data = data.dropna()

# Check columns
print(data.columns)

# Assuming:
# Text column = 'clean_text'
# Sentiment column = 'category'

# ===============================
# 4. Text Cleaning Function
# ===============================
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)        # Remove URLs
    text = re.sub(r'@\w+', '', text)           # Remove mentions
    text = re.sub(r'[^a-zA-Z\s]', '', text)    # Remove special chars
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

data['cleaned'] = data['clean_text'].apply(clean_text)

# ===============================
# 5. Feature Engineering (TF-IDF)
# ===============================
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['cleaned'])
y = data['category']

# ===============================
# 6. Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# 7. Train Naive Bayes Model
# ===============================
model = MultinomialNB()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# ===============================
# 8. Evaluation
# ===============================
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
plt.figure()
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ===============================
# 9. Sentiment Distribution Plot
# ===============================
plt.figure()
data['category'].value_counts().plot(kind='bar')
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()

# ===============================
# 10. WordCloud
# ===============================
all_words = " ".join(data['cleaned'])
wordcloud = WordCloud(width=800, height=400).generate(all_words)

plt.figure()
plt.imshow(wordcloud)
plt.axis("off")
plt.title("Word Cloud")
plt.show()
