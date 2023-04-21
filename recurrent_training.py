import os
from sqlite3 import connect, Cursor
import numpy as np
import tensorflow as tf

from variables import train_database_path, recurrent_model_path
from postprocessing import get_sentiment_cols, get_vocabulary, get_one_hot_encoded_training_data

def main():
    db: Cursor
    try:
        db = connect(train_database_path)
        unique_sentiments, _ = get_sentiment_cols(db)
        vocabulary = get_vocabulary(db)
        vocabulary_list: list[str] = list(vocabulary)
        model = generate_model(unique_sentiments, vocabulary)
        X, Y = get_one_hot_encoded_training_data(db, unique_sentiments, vocabulary_list)
        fit_model(model, X, Y)
        save_model(model)
    finally:
        db.close()

def generate_model(unique_sentiments: list[tuple[str]], vocab_length: int) -> tf.keras.models.Sequential:
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(None, None, vocab_length)),
        tf.keras.layers.LSTM(100, return_sequences=True),
        tf.keras.layers.LSTM(len(unique_sentiments) ** 4),
        tf.keras.layers.Dense(len(unique_sentiments))
    ])
    
    if os.path.isfile(recurrent_model_path):
        model.load_weights(recurrent_model_path)
        print('model loaded from disk')

    model.compile(optimizer='Nadam', loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])
    return model

def fit_model(model: tf.keras.Sequential, X: list[np.ndarray], Y: list[list[list[str]]]):
    model.fit(X, Y, batch_size=16, workers=8, use_multiprocessing=True)

def save_model(model: tf.keras.models.Sequential):
    model.save(recurrent_model_path)

if __name__ == '__main__':
    main()