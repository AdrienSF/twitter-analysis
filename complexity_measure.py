import cudf
from cudf import Series
from cuml.feature_extraction.text import TfidfVectorizer


import os, time, psutil
os.environ["CUDA_VISIBLE_DEVICES"]='1'
from helpers import load_tweets

start = time.time()
filename = 'twitter_data/' + os.listdir('twitter_data')[0]

# load
tweets = Series(load_tweets(filename))
loaded_tweets = len(tweets)


load_mem = psutil.Process(os.getpid()).memory_info().rss
load_time = time.time()-start
with open('tokenize_measure.csv', 'a') as f:
    f.write(str(loaded_tweets) + ',' + str(load_time) + ',' + str(load_mem) + '\n')

# token tfidf vectorize
vec = TfidfVectorizer(stop_words='english')
tfidf_matrix = vec.fit_transform(tweets)

# clear str tweets from memory?
tweets = None

vect_mem = psutil.Process(os.getpid()).memory_info().rss
vect_time = time.time()-load_time
with open('vectorize_measure.csv', 'a') as f:
    f.write(str(loaded_tweets) + ',' + str(vect_time) + ',' + str(vect_mem) + ',' + str(tfidf_matrix.shape) + '\n')


