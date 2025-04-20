import pandas as pd
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Load data
df = pd.read_csv(r"C:\Users\Kritika Gulati\genai-pulse\data\processed\cleaned_reddit_genai_posts.csv")

# Define mapping and target industries
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

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Filter for ChatGPT-related posts
df_chatgpt = df[df['keyword'].str.lower().str.contains('chatgpt')]

# Map subreddits to industries
df_chatgpt['mapped_industry'] = df_chatgpt['subreddit'].map(industry_map)

# Drop any that didn’t map
df_chatgpt = df_chatgpt[df_chatgpt['mapped_industry'].isin(target_industries)]

# Combine title and selftext
df_chatgpt['full_text'] = df_chatgpt['title'] + ' ' + df_chatgpt['selftext'].fillna('')

# Assign sentiment label
def get_sentiment(score):
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Apply sentiment analysis
df_chatgpt['sentiment_score'] = df_chatgpt['full_text'].apply(lambda x: sia.polarity_scores(x)['compound'])
df_chatgpt['sentiment'] = df_chatgpt['sentiment_score'].apply(get_sentiment)

# Group and count sentiments
sentiment_counts = df_chatgpt.groupby(['mapped_industry', 'sentiment']).size().unstack(fill_value=0)
sentiment_counts = sentiment_counts[['Positive', 'Neutral', 'Negative']]  # preserve order

# Plot
sentiment_counts.plot(kind='bar', stacked=True, figsize=(10, 6), color=['#90ee90', '#f0e68c', '#f08080'])

plt.title('Reception of ChatGPT Across Industries', fontsize=16)
plt.xlabel('Industry')
plt.ylabel('Number of Mentions')
plt.legend(title='Sentiment')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()