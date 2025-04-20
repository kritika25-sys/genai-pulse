import pandas as pd
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer

df = pd.read_csv(r"C:\Users\Kritika Gulati\genai-pulse\data\processed\cleaned_reddit_genai_posts.csv")

# Mapping subreddit to broader industries
industry_map = {
    "Technology": "IT/Tech",
    "MachineLearning": "IT/Tech",
    "Data": "IT/Tech",
    "Computers": "IT/Tech",
    "ArtificialIntelligence": "IT/Tech",
    "OpenAI": "IT/Tech",
    "DataScience": "IT/Tech",
    "Healthcare": "Healthcare",
    "Education": "Education",
    "Finance": "Finance",
    "Retail": "Retail",
    "Media": "Media & Publishing",
    "Entertainment": "Entertainment",
    "Legal": "Legal"
}


target_industries = ["IT/Tech", "Education", "Legal", "Healthcare"]

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Filter for ChatGPT posts
df_chatgpt = df[df['keyword'].str.lower().str.contains('chatgpt')].copy()

# Map industries
df_chatgpt['mapped_industry'] = df_chatgpt['subreddit'].map(industry_map)

# Keep only relevant industries
df_chatgpt = df_chatgpt[df_chatgpt['mapped_industry'].isin(target_industries)]

# Combine text
df_chatgpt['full_text'] = df_chatgpt['title'] + ' ' + df_chatgpt['selftext'].fillna('')

# Sentiment classification
df_chatgpt['sentiment_score'] = df_chatgpt['full_text'].apply(lambda x: sia.polarity_scores(x)['compound'])
df_chatgpt['sentiment'] = df_chatgpt['sentiment_score'].apply(lambda x: 'Negative' if x <= -0.05 else ('Positive' if x >= 0.05 else 'Neutral'))

# Filter for Negative Sentiment only
negative_df = df_chatgpt[df_chatgpt['sentiment'] == 'Negative']

# Count per industry
neg_counts = negative_df['mapped_industry'].value_counts().reindex(target_industries, fill_value=0)

# Plotting
plt.figure(figsize=(8, 5))
bars = plt.bar(neg_counts.index, neg_counts.values, color='#f08080')

plt.title("Negative Mentions of ChatGPT Across Industries", fontsize=14)
plt.xlabel("Industry")
plt.ylabel("Number of Negative Posts")
plt.xticks(rotation=0)

# Adding value labels on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, int(yval), ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()