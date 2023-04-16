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
possible_sentiments = list(dataset['sentiment'].unique())

def clean_tweet(tweet: str, sentiment: str, possible_sentiments: list[str], to_update_groupby_word_and_sentiment_count: dict[dict[int]] = None, vocabulary: set = None) -> str | tuple[str, dict[dict[int]], set]:
    stemmer = nltk.PorterStemmer()
    words_to_exclude = 'and a is on etc'.split(' ')
    punctuations = list('.,?!$\"&()/')
    for punctuation in punctuations:
        tweet = tweet.replace(punctuation, ' ')
        tweet = tweet.replace('  ', ' ')
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
        
        word = word.removeprefix('#')
        cleaned_tweet += f' {word}'
        cleaned_tweet = cleaned_tweet.removeprefix(' ')
        if to_update_groupby_word_and_sentiment_count is not None and vocabulary is not None:
            vocabulary.add(word)
            if word not in to_update_groupby_word_and_sentiment_count.keys():
                to_update_groupby_word_and_sentiment_count[word] = {}
                for possible_sentiment in possible_sentiments:
                    to_update_groupby_word_and_sentiment_count[word][possible_sentiment] = 0

            to_update_groupby_word_and_sentiment_count[word][sentiment] += 1

    if vocabulary is not None and to_update_groupby_word_and_sentiment_count is not None:
        return (cleaned_tweet, to_update_groupby_word_and_sentiment_count, vocabulary)
    
    return cleaned_tweet

vocabulary = set()
tweets_sentiment = []
cleaned_tweets = []

#                           word|sentiment|count
groupby_word_and_sentiment_count: dict[dict[int]] = {}
i = 0
for tweet, sentiment in zip(dataset['tweet'], dataset['sentiment']):
    tweet: str = tweet
    sentiment: str = sentiment
    
    cleaned_tweet, groupby_word_and_sentiment_count, vocabulary = clean_tweet(tweet, sentiment, possible_sentiments, groupby_word_and_sentiment_count, vocabulary)
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
    cursor.execute('CREATE TABLE words (word_id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT, word TEXT UNIQUE NOT NULL);')
    sentiment_cols = ''
    prefix = '\n\t, '
    for sentiment in unique_sentiments:
        sentiment_cols += f'{prefix}{sentiment.lower()} TEXT NOT NULL'
    sentiment_cols = sentiment_cols.removeprefix(prefix)
    cursor.execute('CREATE TABLE per_word_per_sentiment_count \n(\n\tword_id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT\n\t, {}\n);'.format(sentiment_cols))
    cursor.execute('CREATE TABLE tweets (tweet TEXT NOT NULL, sentiment TEXT NOT NULL);')
    cursor.connection.commit()

def populate_tables(cursor: Cursor, tweets_and_sentiments: list[tuple[str, str]], vocabulary: set, groupby_word_and_sentiment_count: dict[dict[int]]):
    for word in vocabulary:
        cursor.execute('INSERT INTO words (word) VALUES (?)', (word,))
    print(f'Inserted {len(vocabulary)} words into database.')

    tweets_and_sentiments: list[tuple[str, str]] = list(tweets_and_sentiments)
    for tweet_sentiment_tuple in tweets_and_sentiments:
        cursor.execute('INSERT INTO tweets (tweet, sentiment) VALUES (?, ?)', tweet_sentiment_tuple)
    print(f'Inserted {len(tweets_and_sentiments)} tweets into database.')

    cursor.connection.commit()

create_tables(db, unique_sentiments=possible_sentiments)
populate_tables(db, zip(cleaned_tweets, tweets_sentiment), vocabulary, groupby_word_and_sentiment_count)

for tweet, sentiment in zip(cleaned_tweets, tweets_sentiment):
    words = tweet.split(' ')

db.close()
print('Connection closed.')