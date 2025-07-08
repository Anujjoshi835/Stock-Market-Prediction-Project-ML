import praw
from textblob import TextBlob
import constants as ct  # Make sure your file is named constants.py

# Initialize Reddit API client using credentials from constants.py
reddit = praw.Reddit(
    client_id=ct.CLIENT_ID,
    client_secret=ct.CLIENT_SECRET,
    user_agent=ct.USER_AGENT
)

def get_reddit_sentiment(stock_keyword, limit=50):
    """
    Fetch recent Reddit posts mentioning stock_keyword,
    compute sentiment polarity for each post,
    and return sentiment summary.

    Args:
        stock_keyword (str): Stock ticker or company name to search.
        limit (int): Number of posts to fetch (default 50).

    Returns:
        dict: {
            'average_polarity': float,
            'positive_count': int,
            'negative_count': int,
            'neutral_count': int,
            'texts': list of post texts (title + selftext)
        }
    """
    submissions = reddit.subreddit('all').search(stock_keyword, limit=limit, sort='new')

    polarity_sum = 0
    pos, neg, neutral = 0, 0, 0
    texts = []

    for submission in submissions:
        text = submission.title + ' ' + submission.selftext
        texts.append(text)
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        polarity_sum += polarity

        if polarity > 0:
            pos += 1
        elif polarity < 0:
            neg += 1
        else:
            neutral += 1

    count = len(texts)
    avg_polarity = polarity_sum / count if count > 0 else 0

    return {
        'average_polarity': avg_polarity,
        'positive_count': pos,
        'negative_count': neg,
        'neutral_count': neutral,
        'texts': texts
    }
