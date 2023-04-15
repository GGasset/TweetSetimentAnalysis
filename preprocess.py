import os
import numpy as np
import pandas as pd
from sqlite3 import connect, Cursor
import nltk

from variables import cleaned_train_dataset_path, preprocessed_train_dataset_path

if not os.path.isfile(cleaned_train_dataset_path):
    print('First run setup.py')
    quit(1)

dataset = pd.read_csv(cleaned_train_dataset_path)
dataset = dataset[np.logical_not(dataset['sentiment'].isna()) & np.logical_not(dataset['tweet'].isna())]

words_to_exclude = 'and a is on etc'.split(' ')
punctuations = list('.,?!$\"&')
vocabulary = set()
cleaned_tweets = []
tweets_sentiment = []
stemmer = nltk.PorterStemmer()
i = 0
for tweet, sentiment in zip(dataset['tweet'], dataset['sentiment']):
    tweet: str = tweet
    sentiment: str = sentiment
    for punctuation in punctuations:
        tweet.replace(punctuation, ' ')
        tweet.replace('  ', ' ')
    words_in_tweet = tweet.split(' ')

    cleaned_tweet = ''
    for word in words_in_tweet:
        if os.path.islink(word):
            continue
        if word in words_to_exclude or word.startswith('@'):
            continue
        word = stemmer.stem(word)
        word: str = word
        if word in cleaned_tweet:
            continue

        vocabulary.add(word.removeprefix('#'))
        cleaned_tweet += f' {word}'
    cleaned_tweet = cleaned_tweet.removeprefix(' ')
    
    print(cleaned_tweet)
    cleaned_tweets.append(cleaned_tweet)
    tweets_sentiment.append(sentiment)
    i += 1
    print(f'{i} out of {len(dataset["tweet"])} tweets cleaned, added {len(vocabulary)} words to vocabulary', end='\n\n')

final_cleaned_tweets = []
#database_file = open()
for tweet, sentiment in zip(cleaned_tweets, tweets_sentiment):
    words = tweet.split(' ')
    dataset.index.insert()
dataset.to_csv(preprocessed_train_dataset_path)