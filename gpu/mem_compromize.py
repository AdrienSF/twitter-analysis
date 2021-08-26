from datetime import datetime
import cudf
import cuml
import numpy as np
import cupyx
from cuml.feature_extraction.text import HashingVectorizer
from cuml.feature_extraction.text import CountVectorizer
from cuml.decomposition import IncrementalPCA
import cupy as cp
from pca_cupy import PCA
import tensorly as tl
tl.set_backend('cupy')

# from cuml.preprocessing.text.stem import PorterStemmer

from helpers import gtpjoin

import os, gc, sys, random
import regex as re
from collections import OrderedDict
import pickle
import pandas as pd
# from guppy import hpy; h=hpy()

def log(message: str, other=''):
    message = str(message) +' '+ str(other)
    with open('progress.log', 'a') as f:
        f.write(message+'\n')


def shuffle_forward(l):
    order = list(range(len(l))); random.shuffle(order)
    return list(np.array(l)[order]), order

def shuffle_backward(l, order):
    l_out = cp.zeros((l.shape))
    for i, j in enumerate(order):
        l_out[j] = l[i]
    return l_out

def regexchars(line):    
    """ Returns string with alphabetical characters,    basic puncation, or hashtags sign only. (no numbers)"""    
    return re.sub(r"[^a-zA-Z]", " ", line).lower()


log('START')

def save_distribution(filename, run_name, learning_rate=0.0001, n_iter_train=1000, n_iter_test=150, batch_size=30000, beta_0=0.003):
    # cp.random.seed(seed=0)
    log('loading data')
    if '.csv' in filename:
        df = pd.read_csv(filename)
    else:
        with open(filename, 'rb') as f:
            df = pd.DataFrame(pickle.load(f), columns =['date', 'tweet'])


    log("df['tweet']", df['tweet'].shape)
    import random
    all_tweets = list(df['tweet'].values)
    log('cleaning data')
    # remove hashtags and usernames
    # all_tweets = [re.sub("#[A-Za-z0-9_]+","", re.sub("@[A-Za-z0-9_]+","", tweet)) for tweet in all_tweets]
    all_tweets = [gtpjoin(tweet) for tweet in all_tweets]

    # 1m max?
    order = None

    subsample_size = int(1e6)
    if len(all_tweets) > subsample_size:
        log('subsampling...')
        tweets = random.sample(all_tweets, subsample_size)
        order = None
    else:
        log('no subsampling needed')
        tweets = all_tweets
        log('SHUFFLING INPUT')
        # tweets, order = shuffle_forward(tweets)

    tweets = cudf.Series(tweets)
    log("tweets shape", tweets.shape)
    n_samples = len(tweets)


    vec = CountVectorizer(stop_words='english',
                        # preprocessor=stemmer.stem,
                        lowercase = True, # worksto_dense
                        # ngram_range = (1,),
                        max_df = 0.6, # works
                        min_df = 50,
                        max_features=6000)
                        
    log('vectorizing to get vocab...')
    # vec.fit(tweets)

    # vocab = cudf.Series(vec.vocabulary_.to_arrow().to_pylist() + ['chinesevirus'])

    # log('re-vectorizing with updated vocab')
    # vec = CountVectorizer(stop_words='english',
    #                     lowercase = True, # works
    #                     ngram_range = (1,2),
    #                     vocabulary=vocab)


    dtm = tl.tensor(vec.fit_transform(tweets).toarray(), dtype=tl.float64)
    log("dtm shape", dtm.shape)

    log('converting to cupy...')
    # dtm_sent = cupyx.scipy.sparse.csr_matrix(dtm)
    start = datetime.now()
    log("now =", start)
    # log('mem:', h.heap().size)

    gc.collect()
    log('gc dtm')
    # log('mem:', h.heap().size)


    # batch_size = 30000 # increase batch size to 60 thousand
    #          1000000
    verbose = True
    n_topic =  20

    # beta_0=0.003

    # PCA
    pca = IncrementalPCA(n_components = n_topic, batch_size=batch_size, whiten=True)
    gc.collect()
    log('gc')
    # log('mem:', h.heap().size)

    log('getting mean...')
    M1 = dtm.mean(axis=0)
    log('centering, fit pca...')

    centered = dtm - M1
    log('mean of centered:', cp.mean(centered))
    whitened = pca.fit_transform(centered)
    log("now =", datetime.now())


    log("whitened" , whitened.shape)
    if order:
        log('UNSHUFFLING')
        whitened = shuffle_backward(whitened, order)
    with open('convergence_check/whitened/'+run_name + '_whitened.pickle', 'wb') as f:
        pickle.dump(whitened, f)



    from importlib import reload  
    import tlda_final_cupy
    reload(tlda_final_cupy)
    from tlda_final_cupy import TLDA


    now = datetime.now()
    log("now =", now)
    # learning_rate = 0.01 # shrink
    batch_size =240000
    t = TLDA(n_topic,n_senti=1, alpha_0= beta_0, n_iter_train=n_iter_train, n_iter_test=n_iter_test, batch_size=batch_size, # increase train, 2000
            learning_rate=learning_rate)
    now = datetime.now()
    log("now =", now)


    log("t.factors_.shape", t.factors_.shape)



    now = datetime.now()
    log('fitting tlda...')
    log("now =", now)
    t.fit(whitened, verbose=True) # fit whitened wordcounts to get decomposition of M3 through SGD

    # check cumulant gradient
    log("sum factors: ", cp.sum(t.factors_/(t.learning_rate*10/10010)))


    now = datetime.now()
    log("now =", now)
    log('DONE! Extracting and saving topics...')


    # EXTRACT and SAVE TOPICS
    id_map = vec.get_feature_names()
    


    t.factors_ = pca.inverse_transform(t.factors_)  # unwhiten the eigenvectors to get unscaled word-level factors

    ''' 
    Recover alpha_hat from the eigenvalues of M3
    '''  

    eig_vals = cp.array([cp.linalg.norm(k,3) for k in t.factors_ ])
    # normalize beta
    alpha      = cp.power(eig_vals, -2)

    alpha_norm = (alpha / alpha.sum()) * beta_0
    t.alpha_   = alpha_norm
            
    t.predict(whitened,w_mat=True,doc_predict=False)  # normalize the factors 


    # extract topics and correct weights
    probs = t.factors_
    probmaps = []
    for i in range(n_topic):
        ids = probs[i,:].argsort()[:]
        prob_dict = {id_map[word_id]: float(probs[i,word_id]) for word_id in ids}
        probmaps.append(OrderedDict(sorted(prob_dict.items(), key=lambda x: x[1])))

    with open('convergence_check/distributions/'+run_name+'_distribution.pickle', 'wb') as f:
        pickle.dump(probmaps, f)

    log('success, random state:', cp.random.get_random_state())
    # cp.random.get_random_state()

# learning_rate, n_iter_test = float(sys.argv[1]), int(sys.argv[2])
for run_name in ['run_'+str(i) for i in range(10)]:
    # save_distribution('newsgroups.csv', run_name)#, learning_rate, n_iter_test=n_iter_test)
    save_distribution('newsgroups.csv', run_name, n_iter_train=20000, learning_rate=1e-3, n_iter_test=10)#, learning_rate, n_iter_test=n_iter_test)
# save_distribution('newsgroups.csv', 'newsgroups')
 
