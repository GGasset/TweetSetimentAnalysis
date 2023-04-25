# This file contains any function that comes after the db and before of after the model creation and execution
import string
from sqlite3 import Cursor
import numpy as np
import tensorflow as tf

from preprocess import clean_tweet

def get_one_hot_encoded_character_training_data(tweet_sentiment_list: list[tuple[str, str]], unique_sentiments: list[tuple[str]], max_character_count: int):
    ascii_chars = get_ascii_characters()
    X = np.zeros(shape=(len(tweet_sentiment_list), max_character_count, len(ascii_chars)), dtype='uint8')
    Y = np.zeros(shape=(len(tweet_sentiment_list), max_character_count, len(unique_sentiments)), dtype='uint8')
    for i, (tweet, sentiment) in enumerate(tweet_sentiment_list):
        for char_i, char in enumerate(tweet):
            X[i, char_i] = character_to_one_hot_encoding(char, ascii_chars)
        Y[i] = sentiment_to_ndarray(sentiment, unique_sentiments, len(tweet), max_character_count)

    return tf.convert_to_tensor(X, dtype='uint8'), tf.convert_to_tensor(Y, dtype='uint8')

def character_to_one_hot_encoding(character: str, ascii_chars: list[str]) -> np.ndarray[int]:
    output = np.zeros(shape=(len(get_ascii_characters()),), dtype='uint8')
    output[ascii_chars.index(character)] = 1
    return output

def get_ascii_characters() -> list[str]:
    return list(string.printable)

def get_one_hot_encoded_word_training_data(tweet_sentiment_list: list[tuple[str, str]], unique_sentiments: list[tuple[str]], vocabulary: set[str], vocabulary_list: list[str], max_word_count: int) -> tuple[tf.Tensor, tf.Tensor]:
    X: np.ndarray[np.ndarray[np.ndarray[int]]]
    Y: np.ndarray[np.ndarray[np.ndarray[int]]]

    X = np.zeros(shape=(len(tweet_sentiment_list), max_word_count, len(vocabulary),), dtype='uint8')
    Y = np.zeros(shape=(len(tweet_sentiment_list), max_word_count, len(unique_sentiments),), dtype='uint8')
    for i, (tweet, sentiment) in enumerate(tweet_sentiment_list):
        X[i] = tweet_to_one_hot_encoding_list(tweet, vocabulary, vocabulary_list, max_word_count, is_tweet_cleaned=True)
        Y[i] = sentiment_to_ndarray(sentiment, unique_sentiments, len(tweet.split(' ')), max_word_count)
        print(f'Generated {i} out of {len(tweet_sentiment_list)} data points', end='\r')
    print(f'Appended {len(tweet_sentiment_list)} * total_word_count training data points')
    return (tf.convert_to_tensor(X, dtype='uint8'), tf.convert_to_tensor(Y, dtype='uint8'))

def tweet_to_one_hot_encoding_list(tweet: str, vocabulary: set[str], vocabulary_list: list[str], max_word_count: int, is_tweet_cleaned: bool = False) -> np.ndarray[np.ndarray[int]]:
    if not is_tweet_cleaned:
        tweet = clean_tweet(tweet)

    tweet_words = tweet.split(' ')
    one_hot_encoded_tweet = np.zeros(shape=(max_word_count, len(vocabulary_list),), dtype='uint8')
    for i, word in enumerate(tweet_words):
        if word in vocabulary:
            one_hot_encoded_tweet[i][vocabulary_list.index(word)] = 1
            continue

    return one_hot_encoded_tweet

def generate_prediction(db: Cursor, model: tf.keras.models.Sequential, tweet: str, is_cleaned: bool = False) -> dict[float]:
    if not is_cleaned:
        tweet = clean_tweet(tweet)
    
    X = [get_sentiment_list_for_tweet(db, tweet)]
    output = model.predict(X)

    unique_sentiments, _ = get_sentiment_cols(db)
    return output_to_sentiment(output[0], unique_sentiments)

def get_max_word_count(db: Cursor):
    return db.execute('SELECT value FROM important_values WHERE value_name = "max_tweet_word_count"').fetchall()[0][0]

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

def sentiment_to_ndarray(sentiment: str, unique_sentiments: list[str], word_count: int, max_word_count):
    output = np.zeros(shape=(max_word_count, len(unique_sentiments),), dtype='uint8')
    for i in range(word_count):
        output[i][unique_sentiments.index(sentiment)] = 1
    return output

def sentiment_to_output(sentiment: str, unique_sentiments: list[str]) -> list[int]:
   output = [int(possible_sentiment == sentiment) for possible_sentiment in unique_sentiments]
   return output

def output_to_sentiment(model_output: np.ndarray, unique_sentiments: list[tuple[str]]) -> dict[float]:
    output = {}
    for (predicted_value, sentiment) in zip(model_output, unique_sentiments):
        sentiment = sentiment[0]
        output[sentiment] = predicted_value
    return output

def get_vocabulary(db: Cursor) -> set:
    raw_vocab: list[tuple[str]] = db.execute('SELECT word FROM words').fetchall()
    vocabulary = set()
    for word in raw_vocab:
        vocabulary.add(word[0])
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

def get_sentiment_cols(db: Cursor) -> tuple[list[str], str]:
    unique_sentiments = db.execute('SELECT sentiment FROM unique_sentiments').fetchall()
    sentiment_cols = ''
    for sentiment in unique_sentiments:
        sentiment_cols += f', {sentiment[0]}'
    sentiment_cols = sentiment_cols.removeprefix(', ')
    unique_sentiments = [possible_sentiment[0] for possible_sentiment in unique_sentiments]
    return unique_sentiments, sentiment_cols

