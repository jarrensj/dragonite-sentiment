import tweepy
import re
import pickle

from tweepy import OAuthHandler

consumer_key = ''
consumer_secret = ''
access_token = ''
access_secret = ''

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

args = ['dragonite']
api = tweepy.API(auth,timeout=10)

list_tweets = []

query = args[0]

if len(args) == 1:
    for status in tweepy.Cursor(api.search, q=query + " -filter:retweets", lang='en', result_type='recent').items(100):
        list_tweets.append(status.text)

# print(list_tweets)

with open('tfidfmodel.pickle', 'rb') as f:
    vectorizer = pickle.load(f)

with open('classifier.pickle', 'rb') as f:
    clf = pickle.load(f)

total_pos = 0
total_neg = 0

for tweet in list_tweets:
    tweet = re.sub(r"^http://t.co/[a-zA-Z0-9]*\s", " ", tweet)
    tweet = re.sub(r"\s+https://t.co/[a-zA-Z0-9]*\s", " ", tweet)
    tweet = re.sub(r"\s+https://t.co/[a-zA-Z0-9]*$", " ", tweet)
    tweet = tweet.lower()
    tweet = re.sub(r"that's", "that is", tweet)
    tweet = re.sub(r"there's", "there is", tweet)
    tweet = re.sub(r"\W", " ", tweet)
    tweet = re.sub(r"\d", " ", tweet)
    tweet = re.sub(r"s+[a-z]\s+", " ", tweet)
    tweet = re.sub(r"\s+[a=z]$", " ", tweet)
    tweet = re.sub(r"^[a-z]\s+", " ", tweet)
    tweet = re.sub(r"\s+", " ", tweet)
    sent = clf.predict(vectorizer.transform([tweet]).toarray())
    print(tweet,":", sent)
    if sent[0] == 1:
        total_pos += 1
    else:
        total_neg += 1

import matplotlib.pyplot as plt
import numpy as np
objects = ['Positive', 'Negative']
y_pos = np.arange(len(objects))

plt.bar(y_pos, [total_pos, total_neg], alpha = 0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Number')
plt.title('Number of positive and negative tweets')

plt.show()
