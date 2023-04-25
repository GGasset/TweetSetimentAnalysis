import os
import random
from sqlite3 import connect, Cursor
import numpy as np
import tensorflow as tf

from variables import train_database_path, recurrent_model_path
from postprocessing import get_sentiment_cols, get_vocabulary, get_one_hot_encoded_word_training_data

def main():
    db: Cursor
    try:
        db = connect(train_database_path)
        unique_sentiments, _ = get_sentiment_cols(db)
        vocabulary = get_vocabulary(db)
        vocabulary_list: list[str] = list(vocabulary)
        max_word_count, = db.execute('SELECT value FROM important_values WHERE value_name = "max_tweet_word_count"').fetchall()[0]
        tweet_sentiment_list = db.execute('SELECT tweet, sentiment FROM tweets').fetchall()
        random.shuffle(tweet_sentiment_list)
        possible_options = ['one hot', 'one hot character', 'integer character', 'integer word']
        selected = None
        while True:
            print('options:', possible_options)
            selected = input().lower()
            if selected in possible_options:
                break
            else:
                print('Option not in options.')

        if selected == possible_options[0]:
            train_by_one_hot_word_encoding(tweet_sentiment_list, unique_sentiments, vocabulary, vocabulary_list, max_word_count)
        elif selected == possible_options[1]:
            pass
        else:
            raise NotImplementedError('Option not implemented')

    finally:
        db.close()

def train_by_one_hot_character_encoding(tweet_sentiment_list: list[tuple[str]], unique_sentiments: list[str], max_character_count: int):
    pass

def train_by_one_hot_word_encoding(tweet_sentiment_list: list[tuple[str, str]], unique_sentiments: list[str], vocabulary: set, vocabulary_list: list[str], max_word_count: int):
    model = generate_one_hot_word_model(unique_sentiments, len(vocabulary), max_word_count)
    step = 10 ** 1
    stop = (len(tweet_sentiment_list) // step) * step - step
    for i in range(0, stop, step):
        X, Y = get_one_hot_encoded_word_training_data(tweet_sentiment_list[i:i + step], unique_sentiments, vocabulary, vocabulary_list, max_word_count)
        print('Gathered training data for this epoch')
        fit_model(model, X, Y)
        save_model(model)
        print(i, 'out of', stop)
    X, Y = get_one_hot_encoded_word_training_data(tweet_sentiment_list[stop:len(tweet_sentiment_list) - 1], stop, len(tweet_sentiment_list), unique_sentiments, vocabulary, vocabulary_list, max_word_count)
    fit_model(model, X, Y)
    save_model(model)


def generate_one_hot_word_model(unique_sentiments: list[tuple[str]], vocab_length: int, max_word_count: int) -> tf.keras.models.Sequential:
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(max_word_count, vocab_length)),
        tf.keras.layers.Masking(),
        tf.keras.layers.LSTM(len(unique_sentiments) ** 3),
        tf.keras.layers.Dense(len(unique_sentiments))
    ])
    
    if os.path.isfile(recurrent_model_path):
        model.load_weights(recurrent_model_path)
        print('model loaded from disk')

    model.compile(optimizer='Nadam', loss=tf.keras.losses.BinaryCrossentropy())
    return model

def fit_model(model: tf.keras.Sequential, X: tf.Tensor, Y: tf.Tensor):
    model.fit(X, Y, batch_size=1, workers=8, use_multiprocessing=True)

def save_model(model: tf.keras.models.Sequential):
    model.save(recurrent_model_path)

if __name__ == '__main__':
    main()