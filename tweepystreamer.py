
# inspired by https://www.youtube.com/watch?v=1gQ6uG5Ujiw

from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
from google.colab import drive
import json
from tweepy import API
from tweepy import Cursor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
import re
from datetime import date
import datetime
import os
import time

# connect google drive to colab
drive.mount('/content/gdrive')

# create a variable to store the data path on your drive 
path = ''

# load twitter API secrets from an external JSON file 
secrets = json.loads(open(path + 'secrets.json').read())

class Helper():

  def get_hashtag_query_string(self, list_hashtags):
    # build query string containing hashtags from the list_hashtags parameter
    query_hashtags = hashtags[0]
    for hashtag in hashtags[1:]:
      query_hashtags += f" OR {hashtag}"

    # remove retweets from the query result
    query_hashtags += " -filter:retweets"

    return query_hashtags

  def tweet_to_csv(self, all_tweets):
    # save useful information from tweets to csv
    accumulator = []
    list_columns = ['tweet_id', 'date_created', 'source', 'text', 'retweet_count',
                    'favourite_count', 'followers_count', 'friends_count',
                    'location', 'retweeted_status', 'user_id']

    all_tweets = [t._json for t in all_tweets]

    for tweet in all_tweets:
      # print(tweet.get('user'))
      accumulator.append([tweet.get('id_str'), tweet.get('created_at'),
                          tweet.get('source'), tweet.get('full_text'), 
                          tweet.get('retweet_count'), tweet.get('favorite_count'),
                          tweet.get('user').get('followers_count'), 
                          tweet.get('user').get('friends_count'),
                          tweet.get('user').get('location'),
                          tweet.get('retweeted_status'), 
                          tweet.get('user').get('id_str')])

    if os.path.exists(f"{path}tweets_{date.today().strftime('%d-%m-%Y')}.csv"):
      df = pd.read_csv(f"{path}tweets_{date.today().strftime('%d-%m-%Y')}.csv")
      df_plus = pd.DataFrame(data=accumulator, columns=list_columns)

      res_df = pd.concat([df, df_plus]).drop_duplicates().reset_index(drop=True)
      res_df.to_csv(f"{path}tweets_{date.today().strftime('%d-%m-%Y')}.csv", index=False)

    else:
      df = pd.DataFrame(data=accumulator, columns=list_columns)
      df.to_csv(f"{path}tweets_{date.today().strftime('%d-%m-%Y')}.csv", index=False)

class TwitterAuthenticator():
  
  def authenticate_twitter_app(self):
    auth = OAuthHandler(secrets['api_key'], secrets['api_secret_key'])
    auth.set_access_token(secrets['access_token'], secrets['access_token_secrets'])

    return auth

class TwitterClient():

  def __init__(self, twitter_user=None):  # if the user is none, it will get tweets from your timeline
    self.auth = TwitterAuthenticator().authenticate_twitter_app()
    self.twitter_client = API(self.auth, wait_on_rate_limit=True,
                              wait_on_rate_limit_notify=True)
    self.twitter_user = twitter_user
    self.helper = Helper()

  def get_twitter_client_api(self):
    return self.twitter_client

  def get_user_timeline_tweets(self, num_tweets):
    tweets = []
    for tweet in Cursor(self.twitter_client.user_timeline,
                        id=self.twitter_user).items(num_tweets):
      tweets.append(tweet)
    
    return tweets

  def get_friend_list(self, num_friends):
    friend_list = []
    for friend in Cursor(self.twitter_client.friends,
                         id=self.twitter_user).items(num_friends):
      friend_list.append(friend)

    return friend_list

  def get_todays_tweets(self, hashtags):
    query_hashtags = self.helper.get_hashtag_query_string(hashtags)
    all_tweets = []
    
    cursor = Cursor(self.twitter_client.search,
                          q=query_hashtags,
                          lang='en',
                          tweet_mode='extended').pages()

    for i, page in enumerate(cursor): # 15 tweets/iteration
      all_tweets += page
      print(f"{i} --- {all_tweets[-1].created_at} --- {len(all_tweets)}")
      if (datetime.datetime.now() - all_tweets[-1].created_at).days >= 1:
        print(f"End of today's tweets. Bye!")
        print(f"Saving tweets ... iter={i} --- # tweets={len(all_tweets)}")
        self.helper.tweet_to_csv(all_tweets)
        exit()
      
      if i % 1000:
        self.helper.tweet_to_csv(all_tweets)

      if i == 2500:
        exit('30k tweets gathered.')

  def update_tweet_gathered_content(self, tweet_id):
    try:
      tweet = self.twitter_client.get_status(tweet_id)
      print("Tweet found!")
    except:
      print(f"Error! Tweet not found!")
      return [0, 0, 0]
    return [tweet.retweet_count, tweet.favorite_count, tweet.user.followers_count]

# update tweets columns
cols = ['retweet_count', 'favourite_count', 'followers_count']
pass_as_str = lambda x : client.update_tweet_gathered_content(str(x))
client = TwitterClient()

date = datetime.datetime.strptime("19-05-2020", "%d-%m-%Y")
for i in range(0, 4):
  start_time = time.time()
  df = pd.read_csv(f"{path}/ready_to_use/tweets_{date.strftime('%d-%m-%Y')}.csv",
                   lineterminator='\n')
  updated_cols_arr = df['tweet_id'].apply(pass_as_str)
  updated_cols_df = pd.DataFrame.from_records(updated_cols_arr, columns=cols)

  df['retweet_count'] = updated_cols_df.retweet_count
  df['favourite_count'] = updated_cols_df.favourite_count
  df['followers_count'] = updated_cols_df.followers_count

  #save to .csv 
  df.to_csv(f"{path}ready_to_use/updated_csvs/tweets_{date.strftime('%d-%m-%Y')}.csv") 
  print(f"The updated .csv saved to {path}ready_to_use/updated_csvs/tweets_{date.strftime('%d-%m-%Y')}.csv!")

  date += datetime.timedelta(1)
  print(f"Execution time: {(time.time()-start_time)/60} mins")


#class TwitterStreamer():
  """
  Class for streaming and processing live tweets
  """
  def __init__(self):
    self.twitter_authenticator = TwitterAuthenticator()

  def stream_tweets(self, fetched_tweets_filename, hashtag_list):
    #this handles twitter authentication and the connection to Twitter streaming API
    listener = TwitterListener(fetched_tweets_filename)
    auth = self.twitter_authenticator.authenticate_twitter_app()

    #create a twitter stream
    stream = Stream(auth, listener)

    stream.filter(track=hashtag_list)

class TwitterListener(StreamListener): # this class inherits directly from StreamListener class
  """
  Basic listener
  """

  def __init__ (self, fetched_tweets_filename):
    self.fetched_tweets_filename = fetched_tweets_filename

  #inherited method
  def on_data(self, data): # here we deal with the data
    try:
      with open(self.fetched_tweets_filename, 'a') as tf:
        tf.write(data)
        print(data)
      return True
    except BaseException as e:
      print(f"Error on data {str(e)}")

  #inherited method
  def on_error(self, status): # how to deal with errors
    if status == 420:
      # return False in case rate limit occurs
      return False
      
    print(status)

class TweetAnalyzer():
  """
  Functionality for analyzing and categorizing content from tweets
  """
  def tweets_to_data_frame(self, tweets):
    df = pd.DataFrame(data=[tweet.text for tweet in tweets], columns=['Tweets'])

    df['id'] = np.array([tweet.id for tweet in tweets])
    df['len'] = np.array([len(tweet.text) for tweet in tweets])
    df['date'] = np.array([tweet.created_at for tweet in tweets])
    df['source'] = np.array([tweet.source for tweet in tweets])
    df['likes'] = np.array([tweet.favorite_count for tweet in tweets])
    df['retweets'] = np.array([tweet.retweet_count for tweet in tweets])

    return df

  def clean_tweet(self, tweet):
    # removing special characters and hyperlinks
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

  def analyze_sentiment(self, tweet):
    analysis = TextBlob(self.clean_tweet(tweet))

    if analysis.sentiment.polarity > 0:
      return 1
    elif analysis.sentiment.polarity == 0:
      return 0
    else:
      return -1

hashtags=['#corona', '2019-nCov', 'COVID-19', '#COVID19', '#coronavirus', '#lockdown',
          '#covid-19', '#Pandemic']

client = TwitterClient()
client.get_todays_tweets(hashtags)

# analyzer = TweetAnalyzer()
 

# tweets = api.user_timeline(screen_name="realDonaldTrump", count=200)
# df = analyzer.tweets_to_data_frame(tweets)
# df['sentiment_polarity'] = np.array([analyzer.analyze_sentiment(tweet) for tweet in df['Tweets']])
# df

# time_likes = pd.Series(data=df['likes'].values, index = df['date'])
# time_likes.plot(figsize=(16, 4), label='likes', legend=True)

# time_retweets = pd.Series(data=df['retweets'].values, index = df['date'])
# time_retweets.plot(figsize=(16, 4), label='retweets', legend=True)

# plt.show()

df = pd.read_pickle(f"{path}ready_to_use/pickles/cleaned_full_df_17-23.pkl")
df.head(10)