import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned Reddit data
df = pd.read_csv(r"C:\Users\Kritika Gulati\genai-pulse\data\processed\cleaned_reddit_genai_posts.csv")

# Define mapping from subreddit to industry
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

# Map subreddits to industry
if 'subreddit' in df.columns:
    df['industry'] = df['subreddit'].map(industry_map)
else:
    raise ValueError("Subreddit column not found in dataframe")

# Clean model keywords for consistency
df['keyword'] = df['keyword'].str.lower().str.strip()

# Group and count number of mentions
model_industry_counts = df.groupby(['industry', 'keyword']).size().reset_index(name='count')

# Pivot for heatmap
pivot_table = model_industry_counts.pivot(index='industry', columns='keyword', values='count').fillna(0)

# Plot heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_table, annot=True, fmt=".0f", cmap="Blues")
plt.title("Model Mentions Across Industries", fontsize=16)
plt.xlabel("Model")
plt.ylabel("Industry")
plt.tight_layout()
plt.savefig(r"C:\Users\Kritika Gulati\genai-pulse\output\plots\model_mentions_by_industry.png")
plt.show()

# Filter data for ChatGPT
chatgpt_df = df[df['keyword'].str.lower().str.contains('chatgpt')]

# Group by industry and count the posts
industry_counts = chatgpt_df['industry'].value_counts()

# Plot the pie chart
plt.figure(figsize=(8, 8))
wedges, texts = plt.pie(industry_counts, startangle=140, colors=plt.get_cmap('Set3').colors)

# Add a title
plt.title("Distribution of ChatGPT Mentions Across Industries", fontsize=16, fontweight='bold')

# Add a legend
plt.legend(wedges, industry_counts.index, title="Industries", loc="center left", 
bbox_to_anchor=(1, 0.5))

# Display the plot
plt.tight_layout()
plt.savefig(r"C:\Users\Kritika Gulati\genai-pulse\output\plots\model_mentions_chatgpt_pie.png")
plt.show()
