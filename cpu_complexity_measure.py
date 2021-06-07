import warnings
warnings.filterwarnings('ignore')

import sys, os, time, gc
from helpers import load_tweets
from guppy import hpy; h=hpy()
import gensim

files_to_load = int(sys.argv[1])

start = time.time()
filenames = os.listdir('twitter_data')
filenames = [ 'twitter_data/' + name for name in filenames if '.7z' not in name ][:files_to_load]

# load, stem and remove stop words
tweets = load_tweets(filenames, preprocess=True)
loaded_tweets = len(tweets)


load_mem = h.heap().size
load_time = time.time()-start
with open('cpu_load_measure.csv', 'a') as f:
    f.write(str(loaded_tweets) + ',' + str(load_time) + ',' + str(load_mem) + '\n')

# tfidf vectorize
dictionary = gensim.corpora.Dictionary(tweets)
dictionary.filter_extremes(no_below=100, no_above=0.5)

bow_corpus = [dictionary.doc2bow(tweet) for tweet in tweets]
tfidf = gensim.models.TfidfModel(bow_corpus)
tfidf_matrix = tfidf[bow_corpus]

# clear str tweets from memory?
del bow_corpus
del tweets
gc.collect()


vect_mem = h.heap().size
vect_time = time.time() - start - load_time
with open('cpu_tfidfandDict_measure.csv', 'a') as f:
    f.write(str(loaded_tweets) + ',' + str(vect_time) + ',' + str(vect_mem) + ',' + str(tfidf_matrix.shape[1]) + '\n')


