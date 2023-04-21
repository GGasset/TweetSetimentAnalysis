from sqlite3 import connect, Cursor
import numpy as np
import tensorflow as tf

from variables import train_database_path, recurrent_model_path
from postprocessing import get_one_hot_encoded_training_data

def main():
    pass

def generate_model(unique_sentiments: list[tuple[str]], vocab_length: int) -> tf.keras.models.Sequential:
    pass

def fit_model(model: tf.keras.Sequential, X: list[np.ndarray], Y: list[list[list[str]]]):
    pass

if __name__ == '__main__':
    main()