
from google.colab import drive
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import datetime

# connect google drive to colab
drive.mount('/content/gdrive')

# create a variable to store the data path on your drive 
path = ''

# snippet took from: Topic_Modeling+EDA.ipynb
# read pickles with the cleaned tweets

date = datetime.datetime.strptime("17-05-2020", "%d-%m-%Y")
for i in range(0, 10):
  if i == 0: # if it's first iteration
    df = pd.read_pickle(f"{path}ready_to_use/pickles/corpus_{date.strftime('%d-%m-%Y')}.pkl")    
    df_full = pd.read_csv(f"{path}ready_to_use/tweets_{date.strftime('%d-%m-%Y')}.csv",
                          lineterminator='\n')
    df['retweet_count'] = df_full.retweet_count
    df['favourite_count'] = df_full.favourite_count
    df['followers_count'] = df_full.followers_count
    df['user_id'] = df_full.user_id
    
  else:
    df_last = pd.read_pickle(f"{path}ready_to_use/pickles/corpus_{date.strftime('%d-%m-%Y')}.pkl")
    df_last = df_last.reset_index()
    df_full = pd.read_csv(f"{path}ready_to_use/tweets_{date.strftime('%d-%m-%Y')}.csv",
                          lineterminator='\n')
    df_last['retweet_count'] = df_full.retweet_count
    df_last['favourite_count'] = df_full.favourite_count
    df_last['followers_count'] = df_full.followers_count
    df_last['user_id'] = df_full.user_id
    
    df = pd.concat([df, df_last]).reset_index(drop=True)
  
  print(f"df length:{len(df)}---date:{date.strftime('%d-%m-%Y')}")
  date += datetime.timedelta(1)


#count unique user_ids
print(df['user_id'].nunique())

# TextBlob module is built on top of nltk 
# Sentiment Labels: Each word in a corpus is labeled in terms of polarity and subjectivity. 
# A corpus' sentiment is the average of these.
  # Polarity: How positive or negative a word is. -1 is very negative. +1 is very positive.
  # Subjectivity: How subjective, or opinionated a word is. 0 is fact. +1 is very much an opinion.

from textblob import TextBlob

# each tweet is assigned one polarity and one subjectivity score
# Naive Bayes works well with text data

polarity = lambda x: TextBlob(x).sentiment.polarity
subjectivity = lambda x: TextBlob(x).sentiment.subjectivity

df['polarity'] = df['text'].apply(polarity)
df['subjectivity'] = df['text'].apply(subjectivity)

df

# pickle it
df.to_pickle(f"{path}ready_to_use/pickles/final_df_17-26.pkl")

# let's plot the results!
# TODO: extract tweets for some particular users in order to see the evoltion over time: eg: trump

for index in df.index:
  print(f"{index}")
  x = df.polarity.loc[index]
  y = df.subjectivity.loc[index]

  plt.scatter(x, y, color='blue')

plt.title('Sentiment Analysis', fontsize=20)
plt.xlabel('<-- Negative -------- Positive -->', fontsize=15)
plt.ylabel('<-- Facts -------- Opinions -->', fontsize=15)

plt.show()

# our identified topics
topics = {"healthcare": ['health', 'care', 'support', 'line', 'available', 'effort',
                         'emergcy', 'case', 'new', 'death', 'test', 'positi',
                         'testing', 'data', 'report', 'rate',  'nursing', 'daily',
                         'tested', 'confirmed', 'reported', 'increase', 'infectn',
                         'release', 'reporting', 'worldwide', 'low', 'facility', 'staff',
                         'nurse', 'ilness', 'blood'],
          "businesses": ['business', 'crisis', 'care', 'support', 'response', 'community', 'worker',
                         'public', 'service', 'plan', 'company', 'online', 'healthcare',
                         'pay', 'employee', 'market', 'affected'],
          "society": ['plan', 'research', 'job', 'medical', 'economy', 'outbreak',
                      'healthcare', 'group', 'resource', 'line', 'available', 'effort',
                      'educatn', 'emergcy', 'affected',  'minister', 'canada', 'effect', 
                      'world', 'china', 'america', 'leader', 'leadership', 'threat',
                      'vote', 'europe', 'history', 'hero', 'russia', 'uncertainty'],
          "personal_life": ['home', 'family', 'child', 'safe', 'beer', 'prayer'],
          "protection_measures":['virus', 'vaccine', 'spread',  'hospital', 'face', 'died',
                                  'mask', 'study',  'symptom', 'dr', 'stop', 'doctor', 
                                 'disease', 'age', 'person', 'house', 'far', 'infected',
                                  'sick', 'wear', 'problem', 'protect', 'early', 'dying', 'infectn',
                                  'wearing', 'heart',  'eat', 'fear', 'kill', 'victim', 'wrong',
                                 'elderly', 'caused', 'evidce']
          }

# label each tweet
def label_tweet(text):
  max_occurences = 0
  topic_number = -1 # in the beginning the tweet has no topic assigned

  for index, list_words in enumerate(topics.values()):
    number_occurences = 0
    for word in list_words:
      number_occurences += text.count(word)
      
    if number_occurences > max_occurences:
      max_occurences = number_occurences
      topic_number = index

  return topic_number

topic_df = pd.DataFrame(df['text'].apply(label_tweet))

df['Topic_number'] = topic_df['text'] # assign the computed topic column to original df
df

#select only the tweets that have assigned a topic
df_with_topics = df[df.Topic_number != -1]
df_with_topics

# we assume there is only 1 topic/tweet

# calculate the interaction rate/topic - for additional insight regarding the topics
# TODO:test this

# data_stats = df.groupby(['Topic_number']).agg(retweet_mean=('retweet_count','mean'),
#                                               retweet_median=('retweet_count','median'),
#                                               retweet_sum=('retweet_count','sum'),
#                                               followers_mean=('followers_count','mean'),
#                                               followers_median=('followers_count','median'),
#                                               followers_sum=('followers_count','sum'),                                              
#                                               favourite_mean=('favourite_count','mean'),
#                                               favourite_median=('favourite_count','median')
#                                               favourite_sum=('favourite_count','sum'),
#                                               polarity_mean=('polarity','mean')
#                                               polarity_median=('polarity','median'),
#                                               subjectivity_mean=('subjectivity','mean'),
#                                               subjectivity_median=('subjectivity','median'])}

# data_stats_LDA['interaction_rate'] = (data_stats_LDA['retweet_sum']+
#                                       data_stats_LDA['favourite_sum'])/
#                                       data_stats_LDA['followers_sum']

data_stats = df_with_topics.groupby(['Topic_number']).agg(polarity_mean=('polarity','mean'),
                                              polarity_median=('polarity','median'),
                                              subjectivity_mean=('subjectivity','mean'),
                                              subjectivity_median=('subjectivity','median'),
                                              tweets_count = ('tweet_id', 'count'))

data_stats['tweets_count']