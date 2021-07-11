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
from guppy import hpy; h=hpy()


######### IMPORTS ########


# tweet load setup ############
# filenames = ['../data/unzipped/' + name for name in os.listdir('../data/unzipped')]


week = 'startFromWhitened'
# for week in ['test']:


log('load whitened')
whitened_tweet_mat = np.load('whitened_tweet_mat-start2021-07-06.npy')
log('mem after load whitened: ' + str(h.heap().size))

now = datetime.now()
log(now)

del centered_tweet_mat
gc.collect()
log('mem after gc: ' + str(h.heap().size))

n_topic = 500
beta_0=0.003
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
log('mem after tlda fit: ' + str(h.heap().size))
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
log('mem: ' + str(h.heap().size))

log('###################################')
now = datetime.now()
log(now)

log('DONE')