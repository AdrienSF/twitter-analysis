import cudf
import cupy
from cudf import Series
from cuml.feature_extraction.text import TfidfVectorizer, CountVectorizer
import warnings
warnings.filterwarnings('ignore')

import sys, os, time, gc
os.environ["CUDA_VISIBLE_DEVICES"]='1'
from helpers import load_tweets
from guppy import hpy; h=hpy()


files_to_load = int(sys.argv[1])

start = time.time()
filenames = os.listdir('twitter_data')
filenames = [ 'twitter_data/' + name for name in filenames if '.7z' not in name ][:files_to_load]

# load, stem and remove stop words
tweets = Series(load_tweets(filenames, preprocess=True))
loaded_tweets = len(tweets)


load_mem = h.heap().size
load_time = time.time()-start
with open('load_measure.csv', 'a') as f:
    f.write(str(loaded_tweets) + ',' + str(load_time) + ',' + str(load_mem) + '\n')

# token tfidf vectorize
# vec = TfidfVectorizer(stop_words='english')
# tfidf_matrix = vec.fit_transform(tweets)
vec = CountVectorizer(stop_words='english')
bow_matrix = vec.fit_transform(tweets)
cupy.save('bow_matrix', bow_matrix)
# clear str tweets from memory?
del tweets
gc.collect()

exit(0)
vect_mem = h.heap().size
vect_time = time.time() - start - load_time
with open('vectorize_measure.csv', 'a') as f:
    f.write(str(loaded_tweets) + ',' + str(vect_time) + ',' + str(vect_mem) + ',' + str(tfidf_matrix.shape[1]) + '\n')


