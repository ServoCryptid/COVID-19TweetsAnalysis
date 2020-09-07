
import pandas as pd
from google.colab import drive
import re
import string
import pickle
import nltk
import datetime
import time
import os
from textblob import TextBlob

nltk.download('punkt')
nltk.download('wordnet')

# connect google drive to colab
drive.mount('/content/gdrive')

# create a variable to store the data path on your drive 
path = ''

def clean_csv(df):
  '''
  save data in one csv per day 
  if there are many days, save them in separate files and join with other csvs
  if it already exists
  '''
  df['date_created'] = pd.to_datetime(df['date_created'],
                                      format='%a %b %d %H:%M:%S +0000 %Y',
                                      errors='coerce')
  # remove the messed up date rows
  df = df[df['date_created'].notnull()]
  
  #change the date format
  #first save the current format in another column
  df['full_date_created'] = df['date_created']
  df['date_created'] = df['date_created'].dt.strftime("%d-%m-%Y")

  for date, df_slice in df.groupby('date_created'):
    if os.path.exists(f"{path}tweets_{date}.csv"):
      existing_df = pd.read_csv(f"{path}tweets_{date}.csv", lineterminator='\n')
      # change the date format, etc
      existing_df['date_created'] = pd.to_datetime(existing_df['date_created'],
                                                   format='%a %b %d %H:%M:%S +0000 %Y',
                                                   errors='coerce')
      existing_df = existing_df[existing_df['date_created'].notnull()]
      existing_df['full_date_created'] = existing_df['date_created']
      existing_df['date_created'] = existing_df['date_created'].dt.strftime("%d-%m-%Y")

      #save the merged version of .csv
      res_df = pd.concat([df_slice, existing_df]).drop_duplicates().reset_index(drop=True)
      res_df.to_csv(f"{path}/ready_to_use/tweets_{date}.csv", index=False)
      print(f"Merged .csvs saved to {path}/ready_to_use/tweets_{date}.csv")

    else:
      df_slice.to_csv(f"{path}/ready_to_use/tweets_{date}.csv", index=False)
      print(f".csv saved to {path}/ready_to_use/tweets_{date}.csv")

# TODO: Create bi-grams or tri-grams

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

lemmatizer = WordNetLemmatizer()

emoji_pattern = re.compile("["
                            u"\U0001F600-\U0001F64F"  # emoticons
                            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                            u"\U0001F680-\U0001F6FF"  # transport & map symbols
                            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                            u"\U00002500-\U00002BEF"  # chinese char
                            u"\U00002702-\U000027B0"
                            u"\U00002702-\U000027B0"
                            u"\U000024C2-\U0001F251"
                            u"\U0001f926-\U0001f937"
                            u"\U00010000-\U0010ffff"
                            u"\u2640-\u2642"
                            u"\u2600-\u2B55"
                            u"\u200d"
                            u"\u23cf"
                            u"\u23e9"
                            u"\u231a"
                            u"\ufe0f"  # dingbats
                            u"\u3030"
                            "]+", flags=re.UNICODE)

def clean_tweet(text):
  res_text = ""
  correct_text = ""

  # Remove user mentions
  text = re.sub("@[A-Za-z0-9]+", "", text)

  # Remove emojis
  text = emoji_pattern.sub("", text)

  # Make text all lower case
  text = text.lower()

  # Remove punctuation
  text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)

  # Remove words containing numbers
  text = re.sub('\w*\d\w*', '', text)

  #remove white spaces
  text = text.strip()

  # Dealing with typos
  for word in text.split(' '):
    correct_text += " " + str(TextBlob(word).correct())

  # Lemmatization
  for word in word_tokenize(correct_text):
    res_text += " " + lemmatizer.lemmatize(word)

  return res_text

# process .csvs
# Inspo: https://github.com/adashofdata/nlp-in-python-tutorial
date = datetime.datetime.strptime("17-05-2020", "%d-%m-%Y")

for i in range(0, 14): # process the first 14 days of tweets (17-30.05), 10K tweets for each day
  start_exec_time = time.time() # in order for us to see how long it takes to clean the data
  df = pd.read_csv(f'{path}/ready_to_use/tweets_{date.strftime("%d-%m-%Y")}.csv', lineterminator='\n')
  print(f"iteration: {i} --- Dataframe length: {len(df)}")

  # shuffle df 
  df = df.sample(frac=1).reset_index(drop=True) # drop the previous index column

  # use only the first 10K entries
  df_data = df.head(10000).reset_index(drop=True)
  df_extra = df.tail(len(df)-len(df_data)).reset_index(drop=True) 

  # save extra for later use
  df_extra.to_csv(f"{path}/extra_tweets/tweets_{date.strftime('%d-%m-%Y')}.csv", index=False)

  # clean tweet's text 
  data_clean = pd.DataFrame(df_data.text.apply(clean_tweet))
  print(f"Execution time: {(time.time() - start_exec_time)/60} mins---Date:{date.strftime('%d-%m-%Y')}")

  data_clean['tweet_id'] = df_data.tweet_id
  # print(data_clean)

  # pickle it for later use
  data_clean.to_pickle(f'{path}/ready_to_use/pickles/corpus_{date.strftime("%d-%m-%Y")}.pkl')
  
  date += datetime.timedelta(1)


# print(df.columns)

# read pickles with the cleaned tweets

date = datetime.datetime.strptime("17-05-2020", "%d-%m-%Y")

# 14 days requires too much RAM (for the further computings needed), we'll try
# with 10 days
for i in range(0, 10):
  if i == 0: # if it's first iteration
    df = pd.read_pickle(f"{path}/ready_to_use/pickles/corpus_{date.strftime('%d-%m-%Y')}.pkl")
  else:
    df_last = pd.read_pickle(f"{path}/ready_to_use/pickles/corpus_{date.strftime('%d-%m-%Y')}.pkl")
    
    df = pd.concat([df, df_last]).reset_index(drop=True)
    
  print(f"df lenght:{len(df)}---date:{date.strftime('%d-%m-%Y')}")
  date += datetime.timedelta(1)

# remove ha amp wa ... etc - remained from the cleaning step
# Aim: have the most 50 word used be literally word, not gibberish
def remove_words(text):
  text = re.sub("ha", "", text)
  text = re.sub("amp", "", text)
  text = re.sub("wa", "", text)
  text = re.sub("tt", "", text)
  text = re.sub("ve", "", text)
  text = re.sub("wt", "", text)
  text = re.sub("tn", "", text)
  text = re.sub("nt", "", text)
  text = re.sub("tr", "", text)
  text = re.sub("io", "", text)
  text = re.sub("en", "", text)
  
  return text

df_clean = pd.DataFrame(df.text.apply(remove_words))
df_clean['tweet_id'] = df['tweet_id']
df = df_clean

# checkpoint: pickle the cleaned, full dataset for later use; it only contains the tweet_id and text
df.to_pickle(f"{path}/ready_to_use/pickles/cleaned_full_df_17-26.pkl")

df