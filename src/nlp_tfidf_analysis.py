import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"C:\Users\Kritika Gulati\genai-pulse\data\processed\cleaned_reddit_genai_posts.csv")
df.dropna(subset=['title'], inplace=True)

custom_stopwords = ENGLISH_STOP_WORDS.union({'ai', 'model', 'use', 'using', 'language', 'system', 'tool', 'tools',
                       'models', 'new', 'based', 'machine', 'learning', 'used', 'chatgpt', 'llm', 'llms'})

keywords = df['keyword'].unique()

def plot_top_tfidf_terms(df, keyword, top_n=10):
    keyword_data = df[df['keyword']==keyword]
    text_data = keyword_data['title'].to_list()

    if not text_data:
        print(f"No data for keyword: {keyword}")
        return
    
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(2,3), max_df=0.85, min_df=2,
                                 lowercase=True)
    vectorizer.stop_words_=vectorizer.get_stop_words().union(custom_stopwords)
    X = vectorizer.fit_transform(text_data)
    tfidf_scores = zip(vectorizer.get_feature_names_out(), X.sum(axis=0).A1)
    sorted_terms = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)[:top_n]

    terms, scores = zip(*sorted_terms)

    plt.figure(figsize=(10,5))
    sns.barplot(x=scores, y=terms, palette="crest")
    plt.title(f"Top {top_n} Tf-IDF Terms for '{keyword}'")
    plt.xlabel("TF-IDF Score")
    plt.ylabel("Term")
    plt.tight_layout()
    plt.show()

for kw in keywords:
    plot_top_tfidf_terms(df, kw, top_n=20)