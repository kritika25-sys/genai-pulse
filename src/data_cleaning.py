import pandas as pd

df = pd.read_csv(r"C:\Users\Kritika Gulati\genai-pulse\data\raw\reddit_genai_posts.csv")

df.drop_duplicates(subset=["title", "selftext"], inplace=True)

df.dropna(subset=["title", "selftext"], inplace=True)

df["keyword"] = df["keyword"].str.lower()

df.to_csv(r"C:\Users\Kritika Gulati\genai-pulse\data\processed\cleaned_reddit_genai_posts.csv", index = False)

print(f" Cleaned data saved. Final post count: {len(df)}")