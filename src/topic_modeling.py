import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('stopwords')
nltk.download('wordnet')

# Load dataset
file_path = r"C:\Users\Kritika Gulati\genai-pulse\data\processed\cleaned_reddit_genai_posts.csv"
df = pd.read_csv(file_path, parse_dates=["date"])

# Preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2]
    return " ".join(tokens)

def display_topics(model, feature_names, no_top_words=10):
    for topic_idx, topic in enumerate(model.components_):
        print(f"\n🔹 Topic {topic_idx + 1}:")
        print("   " + ", ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

# Parameters
highpoints = ["2023-03", "2023-05", "2023-06", "2024-06", "2024-07", "2024-09", "2025-03"]
num_topics = 4

# Loop through each highpoint
for hp in highpoints:
    print(f"\n\n========== {hp} ==========")
    
    df_month = df[(df["keyword"] == "chatgpt") & (df["date"].dt.strftime("%Y-%m") == hp)].copy()
    docs = (df_month["title"].fillna("") + " " + df_month["selftext"].fillna("")).tolist()
    docs = [doc.strip() for doc in docs if len(doc.strip()) > 20]

    print(f"📌 Total valid posts: {len(docs)}")
    
    if len(docs) < 10:
        print("⚠️ Skipping due to insufficient data.")
        continue

    # Clean and vectorize
    clean_docs = [preprocess(doc) for doc in docs]
    vectorizer = CountVectorizer(max_df=0.9, min_df=2, stop_words='english')
    X = vectorizer.fit_transform(clean_docs)

    if X.shape[0] < num_topics:
        print("⚠️ Not enough documents after vectorization.")
        continue

    lda_model = LatentDirichletAllocation(
        n_components=num_topics,
        max_iter=20,
        learning_method='online',
        random_state=42,
        learning_decay=0.7
    )
    lda_model.fit(X)

    feature_names = vectorizer.get_feature_names_out()
    display_topics(lda_model, feature_names)

# Load data
file_path = r"C:\Users\Kritika Gulati\genai-pulse\data\processed\cleaned_reddit_genai_posts.csv"
df = pd.read_csv(file_path, parse_dates=["date"])

# Filter for ChatGPT-related posts
df_chatgpt = df[df["keyword"] == "chatgpt"].copy()

# Months of interest
highpoints = ["2023-03", "2023-07", "2024-07", "2024-08", "2024-09", "2025-03"]

# Iterate and display titles
for month in highpoints:
    month_df = df_chatgpt[df_chatgpt["date"].dt.strftime("%Y-%m") == month]
    titles = month_df["title"].dropna().tolist()
    
    print(f"\n========== {month} ==========")
    print(f"📌 Total titles: {len(titles)}\n")
    
    for i, title in enumerate(titles, 1):
        print(f"{i}. {title}")
