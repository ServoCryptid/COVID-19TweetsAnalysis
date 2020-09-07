
# %pip install pyLDAvis

import pyLDAvis
import pyLDAvis.gensim

from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from nltk import word_tokenize, pos_tag
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
import pickle
import pandas as pd
import numpy as np
from google.colab import drive
import datetime
from gensim import matutils, models
from gensim import corpora 
import scipy.sparse
import re
import time

# connect google drive to colab
drive.mount('/content/gdrive')
path = ''

def get_LDA_params(dtm):
  ''' 
  Param: 
    - dtm: the document term matrix
  Returns the two objects needed for the LDA model
    - the gensim corpus
    - the id-to-word dictionary
  '''  
  tdm = dtm.transpose() # term document matrix

  # We're going to put the term-document matrix into a new gensim format, from df --> sparse matrix --> gensim corpus
  sparse_counts = scipy.sparse.csr_matrix(tdm)
  corpus = matutils.Sparse2Corpus(sparse_counts)

  # Gensim also requires dictionary of the all terms and their respective location in the term-document matrix
  id2word = dict((v, k) for k,v in cv.vocabulary_.items()) # for every word in the vocabulary that is in the tweets, get the id of it 
  
  return corpus, id2word

def get_document_term_matrix(df, max_feat=1000, min_df=10):
  '''
    Create a document-term matrix from the given dataframe
  '''
  cv = CountVectorizer(max_features=max_feat, min_df=min_df, stop_words='english')
  data_cv = cv.fit_transform(df.text)
  data_dtm = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
  data_dtm.index = df.tweet_id
  
  return data_dtm, cv

def get_LDA_topics(model, num_topics):
  word_dict = {}
  for topic_number in range(num_topics):
    words = model.show_topic(topic_number, topn=100)
    print(f"Words: {words}")
    word_dict[f"Topic_{topic_number}"] = [i[0] for i in words]
  
  for k,v in word_dict.items():
    print(f"\n\n{k}: {v}")

  return pd.DataFrame(word_dict)

#try with 10 days of tweets, see if it doesn't exceeds the RAM limit
df = pd.read_pickle(f"{path}/pickles/cleaned_full_df_17-26.pkl")

#EDA
# Most common words - find these and create word clouds
# we use the most 1k significant words

data_dtm, cv = get_document_term_matrix(df, len(df))
data = data_dtm.transpose()  # it's easier to work with columns, rather than with rows
word_count = data.sum(axis=1).sort_values(ascending=False).to_frame().reset_index()
word_count.columns = ['word', 'count']
print(word_count) # see the most frequent 50 words

# stopwords = set(STOPWORDS)
# wc = WordCloud(stopwords=stopwords, background_color="white", colormap="Dark2",
#                max_font_size=150, random_state=42)

# wc.generate(word_count['word'].head(100))
# plt.imshow(wc)
# plt.axis("off")
    
# plt.show()

# Visualize the topics
def visualize_topics(model, corpus, id2word, cv):
  d = corpora.Dictionary()
  word2id = dict((k, v) for k,v in cv.vocabulary_.items())
  d.id2word = id2word
  d.token2id = word2id

  pyLDAvis.enable_notebook()
  visualization = pyLDAvis.gensim.prepare(model, corpus, d)
  
  return visualization

# create our model
# Topic modeling: L(hidden)D(type of probability distribution)A technique = a tweet contain a mix of probability distrib of various topics(every topics is a mix of probab. distribution of words)
#choosing the number of topics to start with: 30 (top down, bottom up, explore for DM proj)
#TODO: a deep learning model too
# https://radimrehurek.com/gensim/models/ldamodel.html
#Input: document term matrix, # topics, # iterations
  #gensim library is build specifically for topic modeling
#Output: find major topics across tweets
#TODO: https://papers.nips.cc/paper/3902-online-learning-for-latent-dirichlet-allocation.pdf
#TODO hyperparam tuning on number of topics / passes


start_time = time.time()
corpus, id2word = get_LDA_params(data_dtm)

number_topics = 5
number_passes = [50]  # for now i will use only 50 passes
# number_passes = [25, 50, 100]
# alpha = ['asymmetric', 'auto']  # per document topic distrib; asymmetric = 1/topicno
# eta = ['asymmetric', 'auto']  # = beta = per topic word distribution

# due to time constraints, i will choose alpha=auto (bc there are small chunks of text
# and a tweet can't contain a large number of topics); for beta=auto bc
# a topic can contain a different amount of words 
alpha = 'auto'
eta = 'auto'

# Train the model on the corpus.
for number_pass in number_passes:
  lda_model = models.LdaModel(corpus=corpus, 
                              id2word=id2word,
                              num_topics=number_topics,
                              passes=number_pass,
                              alpha=alpha,
                              eta=eta)
  # lda.print_topics()
  res = get_LDA_topics(lda_model, number_topics)
  # save the results 
  res.to_csv(f"{path}/results/LDA_all_{number_topics}_{number_pass}_all_words_10days.csv")
  print(res)
  print(f"Execution time: {(time.time() - start_time)/60} mins")

#https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/#12buildingthetopicmodel
visualize_topics(lda_model, corpus, id2word, cv)

# Compute Coherence Score
#TODO: test why it returns nan
d = corpora.Dictionary()
word2id = dict((k, v) for k,v in cv.vocabulary_.items())
d.id2word = id2word
d.token2id = word2id

coherence_model_lda = models.CoherenceModel(model=lda_model,
                                            texts=list(df['text']),
                                            dictionary=d,
                                            coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print(f"Coherence Score: {coherence_lda}")

#trick: look at nouns only; by default it looks at all words as being the same
# https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html

def get_nouns(text):
  '''
  Given a string of text, tokenize the text and pull out only the nouns.
  '''
  is_noun = lambda pos: pos[:2] == 'NN'
  tokenized_text = word_tokenize(text)
  all_nouns = [word for (word, pos) in pos_tag(tokenized_text) if is_noun(pos)]

  return ' '.join(all_nouns) # create a string from the list of nouns

df = pd.read_pickle(f"{path}/pickles/cleaned_full_df_17-26.pkl")

df_nouns = pd.DataFrame(df.text.apply(get_nouns))
df_nouns['tweet_id'] = df.tweet_id

# create a new document-term matrix
data_dtm_nouns, cv = get_document_term_matrix(df_nouns)
start_time = time.time()
corpus_nouns, id2word_nouns = get_LDA_params(data_dtm_nouns)

number_topics = 5
number_passes = [50] # for now i will use only 50 passes
# number_passes = [25, 50, 100]
# alpha = ['asymmetric', 'auto']  # per document topic distrib; asymmetric = 1/topicno
# eta = ['asymmetric', 'auto']  # = beta = per topic word distribution

# due to time constraints, i will choose alpha=auto (bc there are small chunks of text
# and a tweet can't contain a large number of topics); for beta=auto bc
# a topic can contain a different amount of words 
alpha = 'auto'
eta = 'auto'

# Train the model on the corpus.
for number_pass in number_passes:
  lda_model_nouns = models.LdaModel(corpus=corpus_nouns, 
                              id2word=id2word_nouns,
                              num_topics=number_topics,
                              passes=number_pass,
                              alpha=alpha,
                              eta=eta)
  # lda.print_topics()
  res_nouns = get_LDA_topics(lda_model_nouns, number_topics)
  # save the results
  res_nouns.to_csv(f"{path}/results/LDA_nouns_{number_topics}_{number_pass}_all_words_10days.csv")
  print(res_nouns)
  print(f"Execution time: {(time.time() - start_time)/60} mins")

visualize_topics(lda_model_nouns, corpus_nouns, id2word_nouns, cv)

def get_nouns_adj(text):
  '''
    Given a string of text, tokenize the text and pull out only the nouns and adjectives.
  '''
  is_noun_adj = lambda pos: pos[:2] == 'NN' or pos[:2] == 'JJ'
  tokenized = word_tokenize(text)
  words_of_interest = [word for (word, pos) in pos_tag(tokenized) if is_noun_adj(pos)] 
  return ' '.join(words_of_interest)

df = pd.read_pickle(f"{path}/pickles/cleaned_full_df_17-26.pkl")

df_nouns_adj = pd.DataFrame(df.text.apply(get_nouns_adj))
df_nouns_adj['tweet_id'] = df.tweet_id

data_dtmna, cv = get_document_term_matrix(df_nouns_adj)
corpusna, id2wordna = get_LDA_params(data_dtmna) 

number_topics = 5
number_passes = [50] # for now i will use only 50 passes
# number_passes = [25, 50, 100]
# alpha = ['asymmetric', 'auto']  # per document topic distrib; asymmetric = 1/topicno
# eta = ['asymmetric', 'auto']  # = beta = per topic word distribution

# due to time constraints, i will choose alpha=auto (bc there are small chunks of text
# and a tweet can't contain a large number of topics); for beta=auto bc
# a topic can contain a different amount of words 
alpha = 'auto'
eta = 'auto'

# Train the model on the corpus.
for number_pass in number_passes:
  lda_model_na = models.LdaModel(corpus=corpus_nouns, 
                                id2word=id2word_nouns,
                                num_topics=number_topics,
                                passes=number_pass,
                                alpha=alpha,
                                eta=eta)
  # lda.print_topics()
  res_na = get_LDA_topics(lda_model_na, number_topics)
  # save the results
  res_na.to_csv(f"{path}/results/LDA_na_{number_topics}_{number_pass}_all_words.csv")
  print(res_na)
  print(f"Execution time: {(time.time() - start_time)/60} mins")

visualize_topics(lda_model_na, corpusna, id2wordna, cv)

def get_nouns_adj_verbs(text):
  '''
  Given a string of text, tokenize the text and pull out only the nouns,
  adjectives and verbs
  '''
  is_noun_adj_vb = lambda pos: pos[:2] == 'NN' or pos[:2] == 'JJ' or pos[:2] == 'VB'
  tokenized = word_tokenize(text)
  words_of_interest = [word for (word, pos) in pos_tag(tokenized) if is_noun_adj_vb(pos)] 
  return ' '.join(words_of_interest)

df = pd.read_pickle(f"{path}/pickles/cleaned_full_df_17-26.pkl")

data_nouns_adj_vb = pd.DataFrame(df.text.apply(get_nouns_adj_verbs))
data_nouns_adj_vb['tweet_id'] = df.tweet_id
data_nouns_adj_vb

data_dtmnav, cv = get_document_term_matrix(data_nouns_adj_vb)
corpusnav, id2wordnav = get_LDA_params(data_dtmnav) 

number_topics = 5
number_passes = [50] # for now i will use only 50 passes
# number_passes = [25, 50, 100]
# alpha = ['asymmetric', 'auto']  # per document topic distrib; asymmetric = 1/topicno
# eta = ['asymmetric', 'auto']  # = beta = per topic word distribution

# due to time constraints, i will choose alpha=auto (bc there are small chunks of text
# and a tweet can't contain a large number of topics); for beta=auto bc
# a topic can contain a different amount of words 
alpha = 'auto'
eta = 'auto'

# Train the model on the corpus.
for number_pass in number_passes:
  lda_model_nav = models.LdaModel(corpus=corpusnav, 
                                  id2word=id2wordnav,
                                  num_topics=number_topics,
                                  passes=number_pass,
                                  alpha=alpha,
                                  eta=eta)
  # lda.print_topics()
  res_nav = get_LDA_topics(lda_model_nav, number_topics)
  # save the results 
  res_nav.to_csv(f"{path}/results/LDA_nav_{number_topics}_{number_pass}_all_words_10days.csv")
  print(res_nav)
  print(f"Execution time: {(time.time() - start_time)/60} mins")

visualize_topics(lda_model_nav, corpusnav, id2wordnav, cv)

visualize_topics(lda_model_nav, corpusnav, id2wordnav, cv)

#TODO; try spacy

def get_NMF_topics(model, num_topics):
  word_dict = {}
  for i, topic in enumerate(model.components_):
    print(f"Words: {topic}")
    word_dict[f"Topic_{i}"] = [word for word in id2words[topic.argsort()[-100:]]]
  
  for k,v in word_dict.items():
    print(f"\n\n{k}: {v}")

  return pd.DataFrame(word_dict)

# NMF model: X=WH
# https://mlexplained.com/2017/12/28/a-practical-introduction-to-nmf-nonnegative-matrix-factorization/
# https://scikit-learn.org/stable/modules/decomposition.html#nmf - also good for LDA explanation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

start_time = time.time()
# convert the tweets to a tf-idf weighted term-document matrix
vectorizer = TfidfVectorizer(max_features=2000, min_df=10, stop_words='english')
X = vectorizer.fit_transform(df.text)
id2words = np.array(vectorizer.get_feature_names())

# apply NMF
num_topics = 5
nmf = NMF(n_components=number_topics, solver='mu')
W = nmf.fit_transform(X)

res = get_NMF_topics(nmf, num_topics)
res.to_csv(f"{path}/results/NMF_all_{number_topics}_{number_pass}_all_words_10days.csv")
print(res)
print(f"Execution time: {(time.time() - start_time)/60} mins")

start_time = time.time()

vectorizer = TfidfVectorizer(max_features=2000, min_df=10, stop_words='english')
X = vectorizer.fit_transform(data_nouns_adj_vb.text)
id2words = np.array(vectorizer.get_feature_names())

# apply NMF
num_topics = 5
nmf = NMF(n_components=number_topics, solver='mu')
W = nmf.fit_transform(X)

res = get_NMF_topics(nmf, num_topics)
res.to_csv(f"{path}/results/NMF_nav_{number_topics}_{number_pass}_all_words_10days.csv")
print(res)
print(f"Execution time: {(time.time() - start_time)/60} mins")

# applying LSI model  = SVD for text
from sklearn.feature_extraction.text import TfidfVectorizer

start_time = time.time()

#create a document term matrix with 2000 features - due to limitations of computational power
vectorizer = TfidfVectorizer(max_features=1000, min_df=10, stop_words='english')
X = vectorizer.fit_transform(df.text)

#represent each and every term and document as a vector. We will use the document-term matrix and decompose it into multiple matrices.
from sklearn.decomposition import TruncatedSVD
svd_model = TruncatedSVD(n_components=5, algorithm='randomized', n_iter=100, random_state=122)
svd_model.fit(X)

print(svd_model.components_)
print(f"Execution time: {(time.time()-start_time)/60}")

# get most important 100 words/topic
terms = vectorizer.get_feature_names()

for i, comp in enumerate(svd_model.components_):
    terms_comp = zip(terms, comp)
    sorted_terms = sorted(terms_comp, key= lambda x:x[1], reverse=True)[:100]
    print("Topic "+str(i)+": ")
    words = [t[0] for t in sorted_terms]
    print(words)

# applying LSI model  = SVD for text
start_time = time.time()

#create a document term matrix with 2000 features - due to limitations of computational power
vectorizer = TfidfVectorizer(max_features=2000, min_df=10, stop_words='english')
X = vectorizer.fit_transform(data_nouns_adj_vb.text)

#represent each and every term and document as a vector. We will use the document-term matrix and decompose it into multiple matrices.
from sklearn.decomposition import TruncatedSVD
svd_model = TruncatedSVD(n_components=5, algorithm='randomized', n_iter=100, random_state=122)
svd_model.fit(X)

(svd_model.components_)

# get most important 100 words/topic
terms = vectorizer.get_feature_names()

for i, comp in enumerate(svd_model.components_):
    terms_comp = zip(terms, comp)
    sorted_terms = sorted(terms_comp, key= lambda x:x[1], reverse=True)[:100]
    print("Topic "+str(i)+": ")
    words = [t[0] for t in sorted_terms]
    print(words)

# visualize topics using UMAP (Uniform Manifold Approximation and Projection).
# TODO: after the tweets are labeled to contain a topic
import umap

X_topics = svd_model.fit_transform(X)
embedding = umap.UMAP(n_neighbors=100, min_dist=0.5, random_state=12).fit_transform(X_topics)

plt.figure(figsize=(7,5))
plt.scatter(embedding[:, 0], embedding[:, 1], 
  c = dataset.target,
  s = 10, # size
  edgecolor='none'
)
plt.show()

plt.figure(figsize=(7,5))
plt.scatter(embedding[:, 0], embedding[:, 1], 
  c = df.text, 
  s = 10, # size
  edgecolor='none'
)
plt.show()