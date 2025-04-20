import pandas as pd
from bertopic import BERTopic
from datetime import datetime
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
import string

# Download NLTK stopwords (if not already downloaded)
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Load data
file_path = r"C:\Users\Kritika Gulati\genai-pulse\data\processed\cleaned_reddit_genai_posts.csv"
df = pd.read_csv(file_path, parse_dates=["date"])

# Filter for ChatGPT
df_chatgpt = df[df["keyword"] == "chatgpt"].copy()

# Define highpoints
highpoints = [
    "2023-03", "2023-07", "2024-07", "2024-08", "2024-09", "2025-03"
]
highpoint_topics = {}

# Use a custom embedding model for BERTopic
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load BERTopic without passing stop_words
topic_model = BERTopic(embedding_model=embedding_model, language="english", verbose=True, min_topic_size=5)

# Loop through highpoints
for hp in highpoints:
    df_month = df_chatgpt[df_chatgpt["date"].dt.strftime("%Y-%m") == hp]
    
    # Combine title and selftext, and filter out short posts
    docs = (df_month["title"].fillna("") + " " + df_month["selftext"].fillna("")).tolist()
    docs = [doc.strip() for doc in docs if len(doc.strip()) > 20]  # Filter out very short posts

    # Remove punctuation
    docs = [doc.translate(str.maketrans("", "", string.punctuation)) for doc in docs]

    # Remove stopwords manually
    docs = [" ".join([word for word in doc.split() if word.lower() not in stop_words]) for doc in docs]

    print(f"\n=== {hp} ===")
    print(f"Total posts after filtering: {len(docs)}")
    
    if len(docs) < 5:
        print(f"Skipping {hp} due to insufficient posts.")
        continue
    
    # Apply BERTopic
    topics, probs = topic_model.fit_transform(docs)
    
    topic_info = topic_model.get_topic_info()
    print(topic_info.head())
    
    # Get top topics (excluding noise topics)
    top_topics = topic_info[topic_info.Topic != -1].head(5)
    
    if top_topics.empty:
        print(f"No meaningful topics found for {hp}.")
        continue

    highpoint_topics[hp] = top_topics

    print(f"\nTop Topics for {hp}:")
    for _, row in top_topics.iterrows():
        topic_id = row["Topic"]
        print(f"\nTopic {topic_id} — {row['Name']}")
        for word, _ in topic_model.get_topic(topic_id):
            print(f"  - {word}")