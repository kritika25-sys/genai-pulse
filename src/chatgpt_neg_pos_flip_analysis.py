import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from nltk.corpus import stopwords
import string
import re


# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Load your data (ensure it has a 'sentiment' column and cleaned text)

df = pd.read_csv(r"C:\Users\Kritika Gulati\genai-pulse\data\processed\cleaned_reddit_genai_posts.csv")

# Add a new column 'full_text' combining title and selftext
df['full_text'] = df['title'] + df['selftext'].fillna("")

# Sentiment classification
df['sentiment_score'] = df['full_text'].apply(lambda x: sia.polarity_scores(x)['compound'])
df['sentiment'] = df['sentiment_score'].apply(lambda x: 'Negative' if x <= -0.05 else ('Positive' if x >= 0.05 else 'Neutral'))
df = df[df['sentiment'].isin(['Positive', 'Negative'])].copy()

# Define custom stopwords
base_stopwords = stopwords.words('english')
custom_words = [
    'chatgpt', 'www', 'https', 'said', 'think', 'know', 'really', 'just', 'people',
    'things', 'thing', 'see', 'could', 'going', 'make', 'still', 'many', 'much', 
    'want', 'need', 'does', 'cant', 'every', 'something', 'someone', 'maybe', 
    'take', 'lets', 'way', 'feel', 'work', 'might', 'look', 'even', 'im', 'us', 
    'etc', 'lot', 'new', 'also', 'use', 'using', 'used', 'like', 'ai', 'dont', 'data'
]
stop_words = base_stopwords + custom_words

# Clean text function
def clean_text(text):
    text = re.sub(r"http\S+|www.\S+", "", str(text))
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'[^\w\s]', '', text.lower())
    text = re.sub(r'\s+', ' ', text)
    return text

df['cleaned_text'] = df['title'].fillna('') + ' ' + df['selftext'].fillna('')
df['cleaned_text'] = df['cleaned_text'].apply(clean_text)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words=stop_words, ngram_range=(2, 3), max_features=30)

# Positive TF-IDF
X_pos = vectorizer.fit_transform(df[df['sentiment'] == 'Positive']['cleaned_text'])  
features_pos = vectorizer.get_feature_names_out()
scores_pos = X_pos.sum(axis=0).A1
df_pos = pd.DataFrame({'term': features_pos, 'score': scores_pos}).sort_values(by='score', ascending=False)

# Negative TF-IDF
X_neg = vectorizer.fit_transform(df[df['sentiment'] == 'Negative']['cleaned_text'])  
features_neg = vectorizer.get_feature_names_out()
scores_neg = X_neg.sum(axis=0).A1
df_neg = pd.DataFrame({'term': features_neg, 'score': scores_neg}).sort_values(by='score', ascending=False)

# Plotting function
def plot_tfidf(df, title, color):
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='score', y='term', palette=color)
    plt.title(title, fontsize=14)
    plt.xlabel("TF-IDF Score")
    plt.ylabel("Term")
    plt.tight_layout()
    plt.show()

# Plots
plot_tfidf(df_pos, "Top TF-IDF Terms in Positive Posts", "Greens_r")
plot_tfidf(df_neg, "Top TF-IDF Terms in Negative Posts", "Reds_r")
