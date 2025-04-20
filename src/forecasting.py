import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# 1. Load dataset
file_path = r"C:\Users\Kritika Gulati\genai-pulse\data\processed\cleaned_reddit_genai_posts.csv"
df = pd.read_csv(file_path, parse_dates=["date"])

# 2. Combine title + selftext for analysis
df['text'] = df['title'].fillna('') + " " + df['selftext'].fillna('')

# 3. Apply VADER for sentiment scoring
sia = SentimentIntensityAnalyzer()
df['compound'] = df['text'].apply(lambda x: sia.polarity_scores(x)['compound'])

# 4. Map compound score to sentiment category (optional, for later analysis)
def get_sentiment_label(score):
    if score >= 0.05:
        return 'positive'
    elif score <= -0.05:
        return 'negative'
    else:
        return 'neutral'

df['sentiment'] = df['compound'].apply(get_sentiment_label)

# 5. Filter for a specific GenAI model (e.g., ChatGPT)
df_model = df[df['keyword'].str.lower() == 'chatgpt']

# 6. Monthly average sentiment score
monthly_sentiment = df_model.resample('M', on='date')['compound'].mean().reset_index()
monthly_sentiment.columns = ['ds', 'y']  # Prophet format

# 7. Fit Prophet model
model = Prophet()
model.fit(monthly_sentiment)

# 8. Forecast future sentiment
future = model.make_future_dataframe(periods=6, freq='M')
forecast = model.predict(future)

# 9. Plot result
fig = model.plot(forecast)
plt.title("Monthly Sentiment Forecast for ChatGPT")
plt.xlabel("Date")
plt.ylabel("Average Compound Score")
plt.tight_layout()
df.to_csv(r"C:\Users\Kritika Gulati\genai-pulse\output\plots\chatgpt_forecast.png", index = False)
plt.show()