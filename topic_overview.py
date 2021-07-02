######### IMPORTS ########

import numpy as np
import math
from scipy.stats import gamma
from sklearn.decomposition import IncrementalPCA

import tensorly as tl
from tensorly.cp_tensor import cp_mode_dot
import tensorly.tenalg as tnl
from tensorly.tenalg.core_tenalg import tensor_dot, batched_tensor_dot, outer, inner

# import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from pca import PCA

# Import TensorLy
import tensorly as tl
from tensorly.tenalg import kronecker
from tensorly import norm
from tensorly.decomposition import symmetric_parafac_power_iteration as sym_parafac
from tensorly.tenalg.core_tenalg.tensor_product import batched_tensor_dot
from tensorly.testing import assert_array_equal, assert_array_almost_equal

from tensorly.contrib.sparse.cp_tensor import cp_to_tensor

from tlda_final import TLDA
import cumulant_gradient
import tensor_lda_util as tl_util
## Break down into steps, then re-engineer.

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
from nltk.corpus import stopwords

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer
porter = PorterStemmer()

from helpers import gtp 

        
import gc
from datetime import datetime, date
import random

import scipy

from importlib import reload  
import tlda_final
reload(tlda_final)
from tlda_final import TLDA


import os, time
from helpers import load_tweets, log
# from guppy import hpy; h=hpy()


######### IMPORTS ########


# tweet load setup ############
# filenames = ['../data/unzipped/' + name for name in os.listdir('../data/unzipped')]

dates = {
    'start':
    ['2020-04-23',
    '2020-04-24',
    '2020-04-25',
    '2020-04-26',
    '2020-04-27',
    '2020-04-28',
    '2020-04-29']
    ,
    'mid': 
    ['2020-06-12',
    '2020-06-13',
    '2020-06-14',
    '2020-06-15',
    '2020-06-16',
    '2020-06-17',
    '2020-06-18']
    ,
    'end':
    ['2020-08-11',
    '2020-08-12',
    '2020-08-13',
    '2020-08-14',
    '2020-08-15',
    '2020-08-16',
    '2020-08-17']
    
}

all_filenames = os.listdir('twitter_data')
all_filenames = [ 'twitter_data/'+filename for filename in all_filenames ]




##########################

tweets = []
for week in dates:
    del tweets
    gc.collect()

    week_filenames = [ name for name in all_filenames if any(substring in name for substring in dates[week]) ]

    log('')
    log('START PROCESSING ' + week)

    # measure runtime
    log('loading data...')
    log(datetime.now())
    start = time.time()
    tweets = load_tweets(week_filenames, preprocessor=None, subsample_proportion=.01)
    n_samples = len(tweets)
    log('loaded tweets: ' + str(len(tweets)))
    log('load time: ' + str(time.time()-start))
    # log('mem after load: ' + str(h.heap().size))
    log(datetime.now())


    countvec = CountVectorizer(tokenizer=gtp,
                                    strip_accents = 'unicode', # works
                                    lowercase = True, # works
                                    ngram_range = (1,2),
                                    max_df = 0.4, # works
                                    min_df = int(0.002*n_samples))

    tweet_mat = countvec.fit_transform(tweets)

    sparse_tweet_mat = scipy.sparse.csr_matrix(tweet_mat)

    tweet_tensor = tl.tensor(sparse_tweet_mat.toarray(),dtype=np.float16)

    del sparse_tweet_mat
    gc.collect()

    M1 = tl.mean(tweet_tensor, axis=0)


    centered_tweet_mat = scipy.sparse.csr_matrix(tweet_tensor - M1,dtype=np.float16) #center the data using the first moment 

    gc.collect()


    start = datetime.now()
    log(start)


    batch_size = int(n_samples/20)
    verbose = True
    n_topic =  20

    beta_0=0.003
    log('fitting pca')
    pca = PCA(n_topic, beta_0, 30000)
    pca.fit(centered_tweet_mat) # fits PCA to  data, gives W
    log('whitening')
    whitened_tweet_mat = pca.transform(centered_tweet_mat) # produces a whitened words counts <W,x> for centered data x
    # save whitened for subsequent use
    filename = 'whitened_tweet_mat-' + week + str(date.today()) + '.npy'
    np.save(filename, whitened_tweet_mat, allow_pickle=False)
    log('saved whitened data to ' + filename)

    now = datetime.now()
    log(now)
    pca_time = now - start 


    gc.collect()
    log('pca and whitening time: ' + str(pca_time))


    now = datetime.now()
    log(now)
    log('initializing tlda')
    learning_rate = 0.01 
    batch_size =15000
    t = TLDA(n_topic,n_senti=1, alpha_0= beta_0, n_iter_train=1000, n_iter_test=150, batch_size=batch_size,
            learning_rate=learning_rate)

    now = datetime.now()
    log(now)



    log('training tlda...')
    t.fit(whitened_tweet_mat,verbose=True) # fit whitened wordcounts to get decomposition of M3 through SGD
    log('trained')
    now = datetime.now()
    log(now)




    log('unwhitening eigenvectors to get unscaled word-level factors...')
    t.factors_ = pca.reverse_transform(t.factors_)  # unwhiten the eigenvectors to get unscaled word-level factors

    ''' 
    Recover alpha_hat from the eigenvalues of M3
    '''  
    log('doing other things?################')
    eig_vals = [np.linalg.norm(k,3) for k in t.factors_ ]
    # normalize beta
    alpha      = np.power(eig_vals, -2)
    log(alpha.shape)
    alpha_norm = (alpha / alpha.sum()) * beta_0
    t.alpha_   = alpha_norm
            
    log(alpha_norm)
    log('normalizing the factors?')
    t.predict(whitened_tweet_mat,w_mat=True,doc_predict=False)  # normalize the factors 

    log('###################################')
    now = datetime.now()
    log(now)

    log('saved')

    log('TOPICS:')
    n_top_words=20
    #print(t_n_indices)
    n_sentiments = 1
    for k in range(n_topic*n_sentiments):
        if k ==0:
            t_n_indices   =t.factors_[k,:].argsort()[:-n_top_words - 1:-1]
            top_words_JST = [i for i,v in countvec.vocabulary_.items() if v in t_n_indices]
        else:
            t_n_indices   =t.factors_[k,:].argsort()[:-n_top_words - 1:-1]
            top_words_JST = np.vstack([top_words_JST, [i for i,v in countvec.vocabulary_.items() if v in t_n_indices]])
            log([i for i,v in countvec.vocabulary_.items() if v in t_n_indices])