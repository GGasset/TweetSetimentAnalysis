import os
import numpy as np
import pandas as pd
from sqlite3 import connect, Cursor
import nltk

from variables import cleaned_train_dataset_path, train_database_path

if not os.path.isfile(cleaned_train_dataset_path):
    print('First run setup.py')
    quit(1)

dataset = pd.read_csv(cleaned_train_dataset_path)
#                                                                                                        delete for production
dataset = dataset[np.logical_not(dataset['sentiment'].isna()) & np.logical_not(dataset['tweet'].isna())].drop_duplicates(subset='sentiment')
dataset: pd.DataFrame = dataset
possible_sentiments = dataset['sentiment'].unique()

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
    
    cleaned_tweets.append(cleaned_tweet)
    tweets_sentiment.append(sentiment)
    i += 1
    print(cleaned_tweet)
    print(f'{i} out of {len(dataset["tweet"])} tweets cleaned, added {len(vocabulary)} words to vocabulary', end='\n\n')

final_cleaned_tweets = []
if os.path.isfile(train_database_path):
    os.remove(train_database_path)

database_file = open(train_database_path, mode='w')
database_file.close()

db = connect(train_database_path).cursor()

def create_tables(cursor: Cursor, unique_sentiments: list[str]):
    cursor.execute('CREATE TABLE words (word_id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT, words TEXT UNIQUE NOT NULL)')
    sentiment_cols = ''
    prefix = '\n\t, '
    for sentiment in unique_sentiments:
        sentiment_cols += f'{prefix}{sentiment.lower()} TEXT NOT NULL'
    sentiment_cols = sentiment_cols.removeprefix(prefix)
    cursor.execute('CREATE TABLE per_word_per_sentiment_count \n(\n\tword_id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT\n\t, {}\n);'.format(sentiment_cols))

def populate_tables(cursor: Cursor, per_word_sentiment_count: dict[tuple[str, list[tuple[str, int]]]]):
    pass

create_tables(db, unique_sentiments=possible_sentiments)


for tweet, sentiment in zip(cleaned_tweets, tweets_sentiment):
    words = tweet.split(' ')

db.close()
print('Connection closed.')