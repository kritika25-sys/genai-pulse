import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Initialize SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Load data
df = pd.read_csv(r"C:\Users\Kritika Gulati\genai-pulse\data\processed\cleaned_reddit_genai_posts.csv")

def plot_sentiment_trends(df, keywords, save_path):
    """
    Plots the sentiment trend for the specified models (keywords).
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the Reddit posts data.
    keywords (list): A list of model/keyword names whose sentiment trends need to be plotted.
    """
    # Add a new column 'full_text' combining title and selftext
    df['full_text'] = df['title'] + df['selftext'].fillna("")

    # Perform sentiment analysis and create a 'sentiment_score' column
    df['sentiment_score'] = df['full_text'].apply(lambda x: sia.polarity_scores(str(x))['compound'])

    # Extract year and month from 'date' for monthly aggregation
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.to_period('M').astype(str)

    # Group by month and keyword to get the average sentiment score
    monthly_sentiment = df.groupby(['month', 'keyword'])['sentiment_score'].mean().reset_index()

    # Convert 'month' to datetime for plotting
    monthly_sentiment['month'] = pd.to_datetime(monthly_sentiment['month'])

    # Loop through keywords and plot each one
    for keyword in keywords:
        # Filter data for the current keyword
        keyword_data = monthly_sentiment[monthly_sentiment['keyword'].str.lower() == keyword.lower()]

        # Plot the sentiment trend for the current keyword
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=keyword_data, x='month', y='sentiment_score', label=keyword, linewidth=2)
        plt.title(f"Monthly Sentiment Trend for {keyword}", fontsize=14)
        plt.xlabel('Month')
        plt.ylabel('Average Sentiment Score')
        plt.xticks(rotation=45)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{save_path}/{keyword}_sentiment_trend.png")
        plt.show()

# Example Usage:
# Plot sentiment trends for all keywords
keywords = ["ChatGPT", "Claude", "Copilot", "Sora", "Gemini", "Meta AI"]
save_path = r"C:\Users\Kritika Gulati\genai-pulse\output\plots"
plot_sentiment_trends(df, keywords, save_path)
