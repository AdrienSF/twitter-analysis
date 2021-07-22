# import nvcategory
# import nvstrings
from datetime import datetime
import cudf
import cuml
# import numpy as np
import cupyx
from cuml.feature_extraction.text import HashingVectorizer
from cuml.feature_extraction.text import CountVectorizer
# from cuml import PCA
# from cuml.decomposition import PCA
from cuml.decomposition import IncrementalPCA
import cupy as cp
from pca_cupy import PCA
import tensorly as tl
tl.set_backend('cupy')



# load tweets
import os, gc
import pickle
import pandas as pd
# from helpers import load_tweets
# all_tweets = load_tweets(filenames, preprocessor=None)


print('loading pickle')
filename = '../clean_data/2020-04-22_23-55-53--2020-04-29_23-55-53.pickle'
with open(filename, 'rb') as f:
    df = pd.DataFrame(pickle.load(f), columns =['date', 'tweet'])




print("df['tweet']", df['tweet'].shape)
print('subsanmpling...')
import random
# use only 100000 tweets
all_tweets = list(df['tweet'].values)
tweets = random.sample(all_tweets, 1000000) #mem runs out at 1M
# tweets = all_tweets

tweets = cudf.Series(tweets)
print("tweets shape", tweets.shape)
n_samples = len(tweets)


vec = CountVectorizer(stop_words='english',
                     lowercase = True, # works
                    ngram_range = (1,2),
                    max_df = 0.5, # works
                    min_df = 100,
                    max_features=6000)
                    
print('vectorizing...')

dtm = vec.fit_transform(tweets)
print('converting to cupy...')
dtm_sent = cupyx.scipy.sparse.csr_matrix(dtm)
del dtm
gc.collect()
print('getting mean...')
M1 = dtm_sent.mean(axis=0)
print('centering...')
centered = []
frac = int(dtm.shape[0]/500)
for i in range(501):
    centered.append(cupyx.scipy.sparse.csr_matrix(dtm_sent[i*frac:(i+1)*frac] - M1)) #mem spike

del dtm
gc.collect()

x_cent = cupyx.scipy.sparse.vstack(centered, format='csr')
 



from cuml.decomposition import IncrementalPCA
start = datetime.now()
print("now =", start)


batch_size = 30000
verbose = True
n_topic =  20

beta_0=0.003

pca = IncrementalPCA(n_components = n_topic, batch_size=batch_size, whiten=True)
print('fitting pca...')
pca.fit(x_cent) # fits PCA to  data, gives W
print('pca transforming data...')
x_whit = pca.transform(x_cent) # produces a whitened words counts <W,x> for centered data x
now = datetime.now()
print("now =", now)
pca_time = now - start 

print("x_whit", x_whit.shape)



from importlib import reload  
import tlda_final_cupy
reload(tlda_final_cupy)
from tlda_final_cupy import TLDA


now = datetime.now()
print("now =", now)
learning_rate = 0.01 
batch_size =240000
t = TLDA(n_topic,n_senti=1, alpha_0= beta_0, n_iter_train=1000, n_iter_test=150, batch_size=batch_size,
         learning_rate=learning_rate)
now = datetime.now()
print("now =", now)


print("t.factors_.shape", t.factors_.shape)



now = datetime.now()
print('fitting tlda...')
print("now =", now)
t.fit(x_whit,verbose=True) # fit whitened wordcounts to get decomposition of M3 through SGD

now = datetime.now()
print("now =", now)
print('DONE!')