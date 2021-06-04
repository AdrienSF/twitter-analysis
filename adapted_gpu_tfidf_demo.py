import os
os.environ["CUDA_VISIBLE_DEVICES"]='1'

import cudf
import dask_cudf
import cupy as cp
from cudf import Series
from dask.distributed import Client
from dask_cuda import LocalCUDACluster
from cuml.dask.common import to_sparse_dask_array
from cuml.feature_extraction.text import CountVectorizer
from cuml.dask.feature_extraction.text import TfidfTransformer

import numpy as np
# import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from helpers import load_tweets # dask_cudf and cudf json support is insufficient (complex dicts like tweets aren't really suitable for dataframes, but I was hoping I could get a Series() of tweets)


if __name__ == '__main__': # need to wrap everything in this or it breaks due to multiprocessing issues

    print('creating cuda cluster...')
    # Create a local CUDA cluster
    cluster = LocalCUDACluster()
    client = Client(cluster)


    print('getting filenames...')
    filename = 'twitter_data/' + os.listdir('twitter_data')[0]
    print('loading tweets...')
    tweets = dask_cudf.from_cudf(Series(load_tweets([filename])), npartitions=2) # throws NotImplementedError, I give up on using dask_cudf for now, I find it really difficult to find documentation, and it seems unfinished.



    cv = CountVectorizer(stop_words='english') # no dask analog
    print('count vectorizing...')
    xformed = cv.fit_transform(tweets).astype(cp.float32)
    print('converting to dask sparse array...')
    X = to_sparse_dask_array(xformed, client)



    mutli_gpu_transformer = TfidfTransformer()
    print('multi-gpu transforming to tf-idf...')
    X_transormed = mutli_gpu_transformer.fit_transform(X)
    X_transormed.compute_chunk_sizes()


    print('tfidf matrix shape:', X_transormed.shape)

# before sparse dask array, intern strings! stemmed strings will hav lots of repreats
# load, stem, tokenize (intern str tokens)
# Count or tfidf vectorize to sparse matrix: vec.fit_transform(tweets).astype(cp.float32)
# then, lda on gpu ... ? how?

# plot mem of tokenized tweets in function of number of tweets, intern vs non-intern (save runtimes! + wight of unfiltered input tweets)
# plot mem of sparse tfidf matrix in function of number of tweets (measure at roughly logorithmic intervals?) (save runtimes! + wight of unfiltered input tweets)
# (have rows*cols*floatmem as comparison if using non sparse mat)



# dask counterpart of cudf where docs? searching "dask" in cuml docs shows all the dask functions, but the same search in the cudf docs shows only 3 results (none relating to specific classes or functions).

# is there an available version of tensorly that already has some NLP functionality implemented?

# hint I want access to that cluster I heard of...
# the cluster I have access to has old CUDA (so I can only use old RAPIDS), but I've asked the system administrator to update so I'll be able to use a newer veraion eventually.