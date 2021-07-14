# a monstruous amount of imports
import gc
import os
import random
import time
import pickle
from datetime import date, datetime

import nltk
import scipy
import tensorly as tl
import numpy as np
from guppy import hpy; h=hpy()
from nltk import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from tensorly import norm
from tensorly.contrib.sparse.cp_tensor import cp_to_tensor
from tensorly.decomposition import symmetric_parafac_power_iteration as sym_parafac
from tensorly.tenalg import kronecker
from tensorly.cp_tensor import cp_mode_dot
import tensorly.tenalg as tnl
from tensorly.tenalg.core_tenalg import tensor_dot, batched_tensor_dot, outer, inner
from tensorly.tenalg.core_tenalg.tensor_product import batched_tensor_dot
from tensorly.testing import assert_array_almost_equal, assert_array_equal
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import IncrementalPCA
from scipy.stats import gamma

import cumulant_gradient
import tensor_lda_util as tl_util
import tlda_final
from helpers import log, gtp
from tlda_final import TLDA
from pca import PCA


## nltk downloads

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

porter = PorterStemmer()

######### IMPORTS ########

def save_tlda(filenames: list, n_topics: int, run_name: str, vocab=None, beta_0=.003, learning_rate=0.01, subsample_proportion=1):

    log('')
    log('STARTING RUN: ' + run_name)

    # LOAD
    log('loading data...')
    log(datetime.now())
    start = time.time()
    tweets = []
    for name in filenames:
        with open(name, 'rb') as f:
            tweets = tweets + [t[1] for t in pickle.load(f)]
    n_samples = len(tweets)
    log('loaded tweets: ' + str(len(tweets)))
    log('load time: ' + str(time.time()-start))
    log('mem after load: ' + str(h.heap().size))
    tweets = random.sample(tweets, int(n_samples*subsample_proportion))
    log('sampled tweets: ' + str(len(tweets)))
    gc.collect()
    log('mem after gc: ' + str(h.heap().size))
    log(datetime.now())

    # VECTORIZE 
    log('vectorizing (generating tweet mat)...')
    # countvec = TfidfVectorizer(tokenizer=gtp,
    if vocab:
        countvec = CountVectorizer(tokenizer=gtp,
                                        strip_accents = 'unicode', # works
                                        lowercase = True, # works
                                        ngram_range = (1,2),
                                        # max_df = 0.5, # works
                                        # min_df = 100,
                                        # max_features=1000)
                                        vocabulary=vocab)
    else:
        countvec = CountVectorizer(tokenizer=gtp,
                                        strip_accents = 'unicode', # works
                                        lowercase = True, # works
                                        ngram_range = (1,2),
                                        max_df = 0.5, # works
                                        min_df = 100,
                                        max_features=1000)

    tweet_mat = countvec.fit_transform(tweets)
    log('mem after gen tweet_mat: ' + str(h.heap().size))
    log('dict size: ' + str(len(list(countvec.vocabulary_.items()))))
    # save word-id dictionary
    id_map = np.array(countvec.get_feature_names())
    np.save(run_name + '_id-word-map_' + str(date.today())+'.npy', id_map)
    log('saved id-word map size: ' + str(id_map.shape))

    # DATA MANIPULATION
    # center data and change data structure
    sparse_tweet_mat = scipy.sparse.csr_matrix(tweet_mat)
    log('mem after gen sparse_tweet_mat: ' + str(h.heap().size))
    del tweet_mat
    gc.collect()
    log('mem after gc: ' + str(h.heap().size))
    tweet_tensor = tl.tensor(sparse_tweet_mat.toarray(),dtype=np.float16)
    log('mem after gen tweet_tensor: ' + str(h.heap().size))
    del sparse_tweet_mat
    gc.collect()
    log('mem after gc: ' + str(h.heap().size))
    M1 = tl.mean(tweet_tensor, axis=0)
    centered_tweet_mat = scipy.sparse.csr_matrix(tweet_tensor - M1,dtype=np.float16) #PEAK MEM USAGE 
    log('mem after gen centered_tweet_mat: ' + str(h.heap().size))
    del tweet_tensor
    gc.collect()
    log('mem after gc: ' + str(h.heap().size))

    # PCA
    start = datetime.now()
    log(start)
    batch_size = int(n_samples/20)
    verbose = True
    log('fitting pca')
    pca = PCA(n_topics, beta_0, 30000)
    pca.fit(centered_tweet_mat) # fits PCA to  data, gives W
    # save pca to file (to have inverse transform)
    with open(run_name + '_trained_PCA_' + str(date.today())+'.pickle', 'wb') as f:
        pickle.dump(pca, f)

    log('whitening')
    whitened_tweet_mat = pca.transform(centered_tweet_mat) # produces a whitened words counts <W,x> for centered data x
    # save whitened for subsequent use
    filename = run_name + '_whitened_tweet_mat_' + str(date.today()) + '.npy'
    np.save(filename, whitened_tweet_mat, allow_pickle=False)
    log('saved whitened data to ' + filename)
    log('mem after whitening: ' + str(h.heap().size))
    now = datetime.now()
    log(now)
    pca_time = now - start 
    del centered_tweet_mat
    gc.collect()
    log('pca and whitening time: ' + str(pca_time))
    log('mem after gc: ' + str(h.heap().size))


    # TLDA
    now = datetime.now()
    log(now)
    log('initializing tlda')
    batch_size =15000
    t = TLDA(n_topics,n_senti=1, alpha_0= beta_0, n_iter_train=1000, n_iter_test=150, batch_size=batch_size,
            learning_rate=learning_rate)

    now = datetime.now()
    log(now)
    log('training tlda...')
    t.fit(whitened_tweet_mat,verbose=True) # fit whitened wordcounts to get decomposition of M3 through SGD
    log('trained')
    log('mem after tlda fit: ' + str(h.heap().size))
    now = datetime.now()
    log(now)
    # save trained tlda to file
    with open(run_name + '_trained_TLDA_' + str(date.today())+'.pickle', 'wb') as f:
        pickle.dump(t, f)




    # EXTRACT TOPICS (just to print an overview to the log)
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
    log('mem: ' + str(h.heap().size))

    log('###################################')
    now = datetime.now()
    log(now)

    log('saved')

    log('TOPICS:')
    n_top_words=20
    #print(t_n_indices)
    n_sentiments = 1
    for k in range(n_topics*n_sentiments):
        if k ==0:
            t_n_indices   =t.factors_[k,:].argsort()[:-n_top_words - 1:-1]
            top_words_JST = [i for i,v in countvec.vocabulary_.items() if v in t_n_indices]
        else:
            t_n_indices   =t.factors_[k,:].argsort()[:-n_top_words - 1:-1]
            top_words_JST = np.vstack([top_words_JST, [i for i,v in countvec.vocabulary_.items() if v in t_n_indices]])
            log([i for i,v in countvec.vocabulary_.items() if v in t_n_indices])



# vocab = list(np.load('bigVocabCountVec_id-word-map_2021-07-12.npy')) + gtp('#ChineseVirus')
save_tlda(['clean_data/2020-06-18_00-33-05--2020-06-25_19-36-09.pickle'], n_topics=20, vocab=None, run_name='midCountVec', subsample_proportion=.1)
save_tlda(['clean_data/2020-08-14_21-46-37--2020-08-17_17-59-01.pickle'], n_topics=20, vocab=None, run_name='endCountVec', subsample_proportion=.1)
