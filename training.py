import os
from sqlite3 import connect, Cursor
import tensorflow as tf
from variables import train_database_path
from preprocess import clean_tweet
from postprocessing import get_sentiment_cols, generate_training_data, output_to_sentiment

def main():
    if not os.path.isfile(train_database_path):
        print('First execute preprocess.py')
        quit(1)

    test_tweet = 'I just found out something interesting about me!'

    db = connect(train_database_path)
    try:
        unique_sentiments, sentiment_cols = get_sentiment_cols(db)
        model = generate_model(unique_sentiments)
    finally:
        db.close()

def generate_model(unique_sentiments: list[str]) -> tf.keras.models.Sequential:
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(len(unique_sentiments), activation='sigmoid', input_shape=(len(unique_sentiments),))
    ])

    model.compile(optimizer='nadam', loss=tf.keras.losses.BinaryCrossentropy(), metrics=tf.keras.metrics.BinaryAccuracy())
    return model

def train_model(model: tf.keras.models.Sequential(), db: Cursor):
    pass
    #tweets = db.

if __name__ == '__main__':
    main()