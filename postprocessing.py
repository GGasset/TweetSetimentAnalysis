# This file contains any function that comes after the db and before of after the model creation and execution

from sqlite3 import Cursor
import tensorflow as tf

from preprocess import clean_tweet

def generate_training_data(db: Cursor, unique_sentiments: list[str], sentiment_cols: str) -> tf.Tensor:
    X: list[list[int]] = []
    Y: list[list[int]] = []

    tweet_sentiment_zip = db.execute('SELECT tweet, sentiment FROM tweets').fetchall()
    for tweet, sentiment in tweet_sentiment_zip:
        X.append(get_sentiment_list_for_tweet(db, tweet, unique_sentiments, sentiment_cols))
        Y.append(sentiment_to_output(sentiment, unique_sentiments))

    output = tf.stack([tf.convert_to_tensor(X), tf.convert_to_tensor(Y)], axis=0)
    return output

def sentiment_to_output(sentiment: str, unique_sentiments: list[str]) -> list[int]:
    output = [int(possible_sentiment == sentiment) for possible_sentiment in unique_sentiments]
    return output

def output_to_sentiment(list):
    pass

def get_sentiment_list_for_tweet(db: Cursor, tweet: str, unique_sentiments: list[tuple[str]], sentiment_cols_str: str, is_cleaned: bool = True) -> list[int]:
    if not is_cleaned:
        tweet = clean_tweet(tweet)

    words = tweet.split(' ')

    tweet_sentiments = [0 for i in range(len(unique_sentiments))]
    for word in words:
        word_sentiments_count = db.execute("SELECT {} FROM groupby_word_sentiment_count WHERE word_id = (SELECT word_id FROM words WHERE word = ?)".format(sentiment_cols_str), (word,)).fetchall()[0]
        for sentiment_i, word_sentiment_count in enumerate(word_sentiments_count):
            tweet_sentiments[sentiment_i] += word_sentiment_count
    return tweet_sentiments

def get_sentiment_cols(db: Cursor) -> tuple[list[tuple[str]], str]:
    unique_sentiments = db.execute('SELECT sentiment FROM unique_sentiments').fetchall()
    sentiment_cols = ''
    for sentiment in unique_sentiments:
        sentiment_cols += f', {sentiment[0]}'
    sentiment_cols = sentiment_cols.removeprefix(', ')
    return unique_sentiments, sentiment_cols
