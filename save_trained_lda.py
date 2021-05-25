import pandas as pd
import numpy as np
import gensim
from helpers import lemmatize_stemming, preprocess, compute_coherence_values

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import matplotlib.pyplot as plt
import os

filenames = os.listdir('data/unzipped')
filenames = [ 'data/unzipped/'+filename for filename in filenames ]

import random
indeces = random.sample(range(len(filenames)), 50)
# indeces = [68]


from helpers import load_tweets

documents = load_tweets(np.array(filenames)[indeces])

processed_docs = [ preprocess(doc) for doc in documents ]
dictionary = gensim.corpora.Dictionary(processed_docs)
dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=100000)

bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
tfidf = gensim.models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]

lda_model = gensim.models.LdaMulticore(corpus_tfidf, num_topics=14, id2word=dictionary, passes=10, workers=4, alpha=0.01, eta=.91)

os.makedirs('trained_lda')
lda_model.save('trained_lda/trained_lda')