import os
from sqlite3 import connect
import tensorflow as tf

from variables import train_database_path, model_path
from preprocess import clean_tweet
from training import generate_model
from postprocessing import generate_prediction, get_sentiment_cols, get_sentiment_list_for_tweet, output_to_sentiment

db = connect(train_database_path).cursor()
try:
    model = None
    if os.path.isfile(model_path):
        unique_sentiments, sentiment_cols = get_sentiment_cols(db)
        model = generate_model(unique_sentiments)
    while True:
        print('Input a tweet to clean:', end=' ')
        tweet = clean_tweet(input())
        if tweet == clean_tweet('cancel'):
            quit(0)
        print(output_to_sentiment(get_sentiment_list_for_tweet(db, tweet, unique_sentiments, sentiment_cols), unique_sentiments))
        if model is not None:
            pred = generate_prediction(db, model, tweet, is_cleaned=True)
            sentiment_with_highest_prediction = None
            highest_prediction = -10E30
            for sentiment, prediction in zip(pred.keys(), pred.values()):
                if prediction > highest_prediction:
                    sentiment_with_highest_prediction = sentiment
                    highest_prediction = prediction
            print('Classified as', sentiment)
        print('\n\n--------------------------------------\n\n')
finally:
    db.close()