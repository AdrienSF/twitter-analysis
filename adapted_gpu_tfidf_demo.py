import os
os.environ["CUDA_VISIBLE_DEVICES"]='1'

import cudf
from cudf import Series
from cudf import DataFrame

from cuml.feature_extraction.text import TfidfVectorizer

import numpy as np
# import pandas as pd
import warnings


warnings.filterwarnings('ignore')


from helpers import load_tweets

print('getting filenames...')
filename = os.listdir('twitter_data')[0]
print('loading tweets...')
data = Series(load_tweets([filename]))
print('done')

vec = TfidfVectorizer(stop_words='english')

print('vectorizing...')
tfidf_matrix = vec.fit_transform(tweets)

print('tfidf matrix shape:', tfidf_matrix.shape)
