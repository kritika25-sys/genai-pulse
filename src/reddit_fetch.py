import praw
import pandas as pd
from datetime import datetime
import re

# Initialize Reddit API
reddit = praw.Reddit(
    client_id='F4VMGbXT5MnCE9R23fivCw',
    client_secret='4UfsQosWALfQWdKA7wRBmp9nmjUr2A',
    user_agent='genai-pulse app'
)

# Keywords (original form)
keywords = ["ChatGPT", "Claude", "Copilot", "Sora", "Gemini", "Meta AI"]

# Normalized keyword mapping (remove spaces & lowercase for matching)
normalized_keywords = {re.sub(r"\s+", "", kw).lower(): kw for kw in keywords}

# Compiled pattern for fast matching
pattern = re.compile(r'(' + '|'.join(normalized_keywords.keys()) + r')', re.IGNORECASE)

# Target subreddits
subreddits = ["Technology", "MachineLearning", "Healthcare", "Education", "Finance", "Data","Retail",
              "Media", "Entertainment", "Legal", "ArtificalIntelligence", "OpenAI", "DataScience", 
              "Computers"]

data = []
k = 1

# Start data collection
for subreddit in subreddits:
    for keyword in keywords:
        for post in reddit.subreddit(subreddit).search(keyword, limit=1000, sort="new"):
            full_text = f"{post.title} {post.selftext or ''}"
            text_no_space = re.sub(r"\s+", "", full_text).lower()

            match = pattern.search(text_no_space)
            if match:
                matched_norm = match.group().lower()
                matched_keyword = normalized_keywords.get(matched_norm, "Unknown")

                data.append({
                    "date": datetime.fromtimestamp(post.created_utc),
                    "subreddit": subreddit,
                    "title": post.title,
                    "selftext": post.selftext,
                    "keyword": matched_keyword,
                    "score": post.score,
                    "comments": post.num_comments
                })

                print(f"{k}. Match found in r/{subreddit} for keyword: {matched_keyword}")
                k += 1

# Save to CSV
df = pd.DataFrame(data)
df.to_csv("data/raw/reddit_genai_posts.csv", index=False)
print(f"✅ Saved {len(df)} posts to reddit_genai_posts.csv")
