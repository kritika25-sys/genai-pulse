import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"C:\Users\Kritika Gulati\genai-pulse\data\processed\cleaned_reddit_genai_posts.csv")
sia = SentimentIntensityAnalyzer()

df['full_text'] = df['title']+df['selftext'].fillna("")

df['sentiment_score'] = df['full_text'].apply(lambda x: sia.polarity_scores(str(x))['compound'])

def classify_sentiment(score):
    if score>0.05:
        return 'Positive'
    elif score <-0.05:
        return 'Negative'
    else:
        return 'Neutral'
    
df['sentiment'] = df['sentiment_score'].apply(classify_sentiment)

df.to_csv(r"C:\Users\Kritika Gulati\genai-pulse\data\processed\sentiment_scored_reddit.csv", index = False)

plt.figure(figsize=(10,6))
sns.countplot(data=df, x="keyword", hue='sentiment', palette="Set2")
plt.title("Sentiment Analysis by GenAI keyword (Reddit)")
plt.xlabel("Keyword")
plt.ylabel("Number of Posts")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("output/sentiment_analysis.png")
plt.show()

avg_sentiment = df.groupby('keyword')['sentiment_score'].mean().sort_values()
plt.figure(figsize=(8,5))
sns.barplot(x=avg_sentiment.values, y=avg_sentiment.index, palette="coolwarm")
plt.title("Average Sentiment Score per GenAI Keyword")
plt.xlabel("Average Compound Sentiment Score")
plt.ylabel("Keyword")
plt.tight_layout()
plt.savefig("output/average_compound_sentiment_score.png")
plt.show()

claude_df = df[df["full_text"].str.contains("claude", case=False)]
chatgpt_df = df[df["full_text"].str.contains("chatgpt", case=False)]
claude_sentiment_counts = claude_df["sentiment"].value_counts()
chatgpt_sentiment_counts = chatgpt_df["sentiment"].value_counts()

fig,axs = plt.subplots(1,2, figsize=(12,6))

axs[0].pie(claude_sentiment_counts, labels=claude_sentiment_counts.index, autopct='%1.1f%%',
          startangle=140, colors=['green','grey','red'])
axs[0].set_title("Sentiment Distribution for Claude AI")

axs[1].pie(chatgpt_sentiment_counts, labels=chatgpt_sentiment_counts.index, autopct='%1.1f%%',
          startangle=140, colors=['green','grey','red'])
axs[1].set_title("Sentiment Distribution for ChatGPT")

plt.tight_layout()
plt.show()
plt.savefig("output/pie_charts_chatgpt_claudeAI.png")