import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import dates as mdates
import seaborn as sns
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

df = pd.read_csv(r"C:\Users\Kritika Gulati\genai-pulse\data\processed\cleaned_reddit_genai_posts.csv")
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.to_period('M').astype(str)
sia = SentimentIntensityAnalyzer()

df['full_text'] = df['title']+df['selftext'].fillna("")
df['sentiment_score'] = df['full_text'].apply(lambda x: sia.polarity_scores(str(x))['compound'])

monthly_sentiment = df.groupby(['month', 'keyword'])['sentiment_score'].mean().reset_index()

monthly_sentiment['month'] = pd.to_datetime(monthly_sentiment['month'])
monthly_sentiment = monthly_sentiment.sort_values(by='month')

plt.figure(figsize=(12,6))
sns.lineplot(data = monthly_sentiment, x = 'month', y = 'sentiment_score', hue = 'keyword',
             palette='pastel', linewidth=2.5)

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))

plt.xlim(pd.to_datetime('2022-12-01'), pd.to_datetime('2025-04-01'))

plt.title("Monthly average sentiment score by Model", fontsize=16)
plt.xlabel('Month')
plt.ylabel('Average Compound Sentiment Score')
plt.xticks(rotation=45)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.legend(title='Model')
plt.show()

plt.savefig(r"C:\Users\Kritika Gulati\genai-pulse\output\plots\average_sentiment_over_time.png")

df_chatgpt = df[df['keyword']=='chatGPT']