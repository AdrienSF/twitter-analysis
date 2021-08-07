from datetime import datetime
import cudf
import cuml
# import numpy as np
from cuml.decomposition import IncrementalPCA
import cupyx
from cuml.feature_extraction.text import HashingVectorizer
from cuml.feature_extraction.text import CountVectorizer
from cuml.decomposition import IncrementalPCA
import cupy as cp
from pca_cupy import PCA
import tensorly as tl
tl.set_backend('cupy')



import os, gc
from collections import OrderedDict
import pickle
import pandas as pd
# from guppy import hpy; h=hpy()

def log(message: str, other=''):
    message = str(message) +' '+ str(other)
    with open('progress.log', 'a') as f:
        f.write(message+'\n')


def save_distribution(filename, run_name):
    log('loading pickle')
    with open(filename, 'rb') as f:
        df = pd.DataFrame(pickle.load(f), columns =['date', 'tweet'])

    log("df['tweet']", df['tweet'].shape)
    import random
    all_tweets = list(df['tweet'].values)
    # 1m max
    subsample_size = 100000
    if len(all_tweets) > subsample_size:
        log('subsampling...')
        tweets = random.sample(all_tweets, subsample_size)
    else:
        log('no subsampling needed')
        tweets = all_tweets

    tweets = cudf.Series(tweets)
    log("tweets shape", tweets.shape)
    n_samples = len(tweets)


    vec = CountVectorizer(stop_words='english',
                        lowercase = True, # works
                        ngram_range = (1,2),
                        max_df = 0.5, # works
                        min_df = 100,
                        max_features=1000)
                        
    log('vectorizing to get vocab...')
    vec.fit(tweets)

    vocab = cudf.Series(vec.vocabulary_.to_arrow().to_pylist() + ['chinesevirus'])

    log('re-vectorizing with updated vocab')
    vec = CountVectorizer(stop_words='english',
                        lowercase = True, # works
                        ngram_range = (1,2),
                        vocabulary=vocab)


    dtm = vec.fit_transform(tweets)
    log('converting to cupy...')
    dtm_sent = cupyx.scipy.sparse.csr_matrix(dtm)
    start = datetime.now()
    log("now =", start)
    # log('mem:', h.heap().size)

    del dtm
    gc.collect()
    log('gc dtm')
    # log('mem:', h.heap().size)


    batch_size = 30000 # increase batch size to 60 thousand
    #          1000000
    verbose = True
    n_topic =  20

    beta_0=0.003

    # PCA
    pca = IncrementalPCA(n_components = n_topic, batch_size=batch_size, whiten=True)
    gc.collect()
    log('gc')
    # log('mem:', h.heap().size)

    log('getting mean...')
    M1 = dtm_sent.mean(axis=0)
    log('centering, fit pca...')


    whitened = pca.fit_transform(dtm_sent - M1)
    log("now =", datetime.now())


    log("whitened" , whitened.shape)
    with open(run_name + '_whitened.p', 'wb') as f:
        pickle.dump(whitened, f)



    from importlib import reload  
    import tlda_final_cupy
    reload(tlda_final_cupy)
    from tlda_final_cupy import TLDA


    now = datetime.now()
    log("now =", now)
    learning_rate = 0.01 # shrink
    batch_size =240000
    t = TLDA(n_topic,n_senti=1, alpha_0= beta_0, n_iter_train=1000, n_iter_test=150, batch_size=batch_size, # increase train, 2000
            learning_rate=learning_rate)
    now = datetime.now()
    log("now =", now)


    log("t.factors_.shape", t.factors_.shape)



    now = datetime.now()
    log('fitting tlda...')
    log("now =", now)
    t.fit(whitened, verbose=True) # fit whitened wordcounts to get decomposition of M3 through SGD

    now = datetime.now()
    log("now =", now)
    log('DONE!')


    # EXTRACT and SAVE TOPICS
    id_map = vec.get_feature_names()
    print('id_map:', id_map.shape)
    print('id_map 1000:', id_map[1000])



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
    print('probs:', probs.shape)
    probmaps = []
    for i in range(n_topic):
        ids = probs[i,:].argsort()[:]
        prob_dict = {id_map[word_id]: float(probs[i,word_id]) for word_id in ids}
        probmaps.append(OrderedDict(sorted(prob_dict.items(), key=lambda x: x[1])))

    with open(run_name+'_distribution.p', 'wb') as f:
        pickle.dump(probmaps, f)


for run_name in ['3e4batch_size_run_1', '3e4batch_size_run_2']:
    save_distribution('../week1test_subset.pickle', run_name)
