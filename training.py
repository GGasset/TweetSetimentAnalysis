import os
import time
from sqlite3 import connect, Cursor
import tensorflow as tf

from variables import train_database_path, model_path
from preprocess import clean_tweet
from postprocessing import get_sentiment_cols, generate_training_data

def main():
    print('Warning!\nThis file is obsolete as the network is not able to create accurate predictions. See recurrent_training.py, it features one-hot encoding for tweets and a RNN')
    seconds_of_warning = 10
    for i in range(seconds_of_warning + 1):
        print(f'File will be executed in {seconds_of_warning - i} seconds', end='\r')
        time.sleep(1)
    if not os.path.isfile(train_database_path):
        print('First execute preprocess.py')
        quit(1)

    test_tweet = 'I just found out something interesting about me!'

    db = connect(train_database_path)
    try:
        unique_sentiments, sentiment_cols = get_sentiment_cols(db)
        model = generate_model(unique_sentiments)
        X, Y = generate_training_data(db, unique_sentiments, sentiment_cols)
        train_model(model, X, Y)
        save_model(model)
    finally:
        db.close()

def generate_model(unique_sentiments: list[tuple[str]]) -> tf.keras.models.Sequential:
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(None, len(unique_sentiments))),
        #tf.keras.layers.Dense(len(unique_sentiments), activation='sigmoid', bias_initializer='ones'),
        #tf.keras.layers.Dense(5, activation='sigmoid'),
        tf.keras.layers.Dense(len(unique_sentiments), activation='sigmoid')
    ])

    if os.path.isfile(model_path):
        model.load_weights(model_path)

    model.compile(optimizer='nadam', loss=tf.keras.losses.BinaryCrossentropy(), metrics=[tf.keras.metrics.MeanSquaredError()])
    return model

def train_model(model: tf.keras.models.Sequential, X: tf.Tensor, Y: tf.Tensor):
    model.fit(X, Y, batch_size=10 ** 4, epochs=1, use_multiprocessing=True)

def save_model(model: tf.keras.models.Sequential) -> None:
    model.save(model_path)

if __name__ == '__main__':
    main()