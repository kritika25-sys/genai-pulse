import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\Kritika Gulati\genai-pulse\data\processed\cleaned_reddit_genai_posts.csv")
df['date'] = pd.to_datetime(df['date'])

trend_data = df.groupby([df['date'].dt.to_period('M'), 'keyword']).size().unstack(fill_value=0)

trend_data_smoothed = trend_data.rolling(window=3).mean()

plt.figure(figsize=(12,6))

for keyword in trend_data.columns:
    plt.plot(trend_data_smoothed.index.astype(str), trend_data_smoothed[keyword], label=keyword)

plt.title('Keyword Trends Over Time in Reddit')
plt.xlabel('Month')
plt.ylabel('number of Posts')
plt.xticks(rotation=45)
plt.legend(title='Keywords')
plt.tight_layout()

plt.savefig(r"C:\Users\Kritika Gulati\genai-pulse\data\processed\keyword_trends.png")

plt.show()

keyword_counts = df['keyword'].value_counts()

plt.figure(figsize=(8,8))
plt.pie(keyword_counts, labels=keyword_counts.index, autopct='%1.1f%%', startangle=90,
        colors=plt.cm.Paired.colors)
plt.title('Distribution of keywords in Reddit posts')

plt.savefig(r"C:\Users\Kritika Gulati\genai-pulse\data\processed\keyword_distribution_pie_chart.png")
plt.show()