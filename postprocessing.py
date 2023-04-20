# This file contains any function that comes after the db and before of after the model creation and execution

from sqlite3 import Cursor
import numpy as np
import tensorflow as tf

from preprocess import clean_tweet

def generate_prediction(db: Cursor, model: tf.keras.models.Sequential, tweet: str, is_cleaned: bool = False) -> dict[float]:
    if not is_cleaned:
        tweet = clean_tweet(tweet)
    
    X = get_sentiment_list_for_tweet(db, tweet)
    output = model.predict(X)
    return output_to_sentiment(output)

def generate_training_data(db: Cursor, unique_sentiments: list[str], sentiment_cols: str) -> tuple[tf.Tensor, tf.Tensor]:
    X: list[list[int]] = []
    Y: list[list[int]] = []

    tweet_sentiment_zip = db.execute('SELECT tweet, sentiment FROM tweets').fetchall()
    for i, (tweet, sentiment) in enumerate(tweet_sentiment_zip):
        X.append(get_sentiment_list_for_tweet(db, tweet, unique_sentiments, sentiment_cols))
        Y.append(sentiment_to_output(sentiment, unique_sentiments))
        if not i % 10 ** 4:
            print(f'{i}/{len(tweet_sentiment_zip)} of appended training data')

    return (tf.convert_to_tensor(X, dtype='uint32'), tf.convert_to_tensor(Y, dtype='uint32'))

def sentiment_to_output(sentiment: str, unique_sentiments: list[str]) -> list[int]:
    output = [int(possible_sentiment == sentiment) for possible_sentiment in unique_sentiments]
    return output

def output_to_sentiment(model_output: np.ndarray, unique_sentiments: list[tuple[str]]) -> dict[float]:
    model_output = list(model_output)
    output = {}
    for (predicted_value, sentiment) in zip(model_output, unique_sentiments):
        sentiment = sentiment[0]
        output[sentiment] = predicted_value
    return output

def get_one_hot_encoded_training_data(db: Cursor):
    pass

def tweet_to_one_hot_encoding_list(tweet: str, vocabulary: set[str], is_tweet_cleaned: bool = False) -> np.ndarray:
    if not is_tweet_cleaned:
        tweet = clean_tweet(tweet)

    tweet_words = tweet.split(' ')
    one_hot_encoded_tweet = np.ndarray(shape=(len(tweet_words), len(vocabulary),), dtype='uint8')
    for i, word in enumerate(tweet_words):
        for j, vocab_word in enumerate(vocabulary):
            one_hot_encoded_tweet[i][j] = int(word == vocab_word)



def get_vocabulary(db: Cursor) -> set:
    raw_vocab: list[tuple[str]] = db.execute('SELECT word FROM words').fetchall()
    vocabulary = set()
    for word in raw_vocab:
        vocabulary.add(word)
    return vocabulary

def get_sentiment_list_for_tweet(db: Cursor, tweet: str, unique_sentiments: list[tuple[str]] = None, sentiment_cols_str: str = None, is_cleaned: bool = True) -> list[int]:
    if not is_cleaned:
        tweet = clean_tweet(tweet)

    if unique_sentiments is None or sentiment_cols_str is None:
        unique_sentiments, sentiment_cols_str = get_sentiment_cols(db)

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
