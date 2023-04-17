from preprocess import clean_tweet

# File for demonstration and development help
while True:
    print('Input a tweet to clean:', end=' ')
    tweet = input()
    if tweet == 'cancel':
        quit(0)
    print(clean_tweet(tweet))
    print('\n\n-----------------------------------------------------\n\n')