import os
import json
import random
import pandas as pd
import numpy as np
import gensim
from helpers import lemmatize_stemming, get_preprocessed, compute_coherence_values, load_tweets
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import nltk
nltk.download('wordnet')

filenames = os.listdir('data/unzipped')
filenames = [ 'data/unzipped/'+filename for filename in filenames ]

indeces = random.sample(range(len(filenames)), 50)
# indeces = [68]



documents = load_tweets(np.array(filenames)[indeces])

processed_docs = [ get_preprocessed(doc) for doc in documents ]
dictionary = gensim.corpora.Dictionary(processed_docs)
dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=100000)

bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
tfidf = gensim.models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]

lda_model = gensim.models.LdaMulticore(corpus_tfidf, num_topics=14, id2word=dictionary, passes=10, workers=4, alpha=0.01, eta=.91)

os.makedirs('trained_lda')
lda_model.save('trained_lda/trained_lda')



# make a corpus object that yields data from file instead of loading the whole file.
# to what extent i this possible?
# yield tokenized tweet to dictionary,

class TweetLoader:

    """Iterator that loades, and filters tweets (english non-retweets only)"""

    def __init__(self, filename):
        self.filename = filename

        with open(filename, 'r') as f:
                # add commas between tweets to correct json syntax
            self.tweets = json.loads('['+f.read().replace('}{','},{')+']')

    def __iter__(self):
        return self

    def __next__(self):
        try:
            tweet = next(self.tweets)
        except StopIteration as e:
            raise e

        if 'retweeted_status' not in tweet and tweet['lang'] == 'en':
            return tweet
        else:
            return next(self)


class LazyTokenCorpus:

    """Iterator that loades, preprocesses and tokenizes tweets (lazily)"""

    def __init__(self, filenames):
        self.filenames = filenames
        self.current_file = next(filenames)
        self.token_loader = TweetLoader(self.current_file)

        self.stemmer = SnowballStemmer('english')
        self.lemmatizer = WordNetLemmatizer()

    def __iter__(self):
        return self

    def __next__(self):
        tweet = next(self.token_loader, None)
        if tweet:
            text = tweet['extended_tweet']['full_text'] if 'full_text' in tweet else tweet['text']
            return get_preprocessed(text, self.stemmer, self.lemmatizer)
        else:
            try:
                self.current_file = next(filenames)
            except StopIteration as e:
                raise e

            self.token_loader = TweetLoader(self.current_file)
            return next(self)


dictionary = gensim.corpora.Dictionary(LazyTokenCorpus(filenames))

# 
#  yield dictionary.doc2bow(tokenized tweet) to TfidfModel,
class LazyBowCorpus:
    def __init__(self, dictionary, lazy_token_corpus):
        self.lazy_token_corpus = lazy_token_corpus
        self.dictionary = dictionary

    def __iter__(self):
        return self

    def __next__(self):
        return self.dictionary.doc2bow(next(self.lazy_token_corpus))

tfidf = gensim.models.TfidfModel(LazyBowCorpus(dictionary, LazyTokenCorpus(filenames)))

#  yield tfidfModel[ictionary.doc2bow(tokenized tweet)]
class LazyTfidfCorpus:
    def __init__(self, tfidf_model, lazy_bow_corpus):
        self.tfidf = tfidf_model
        self.lazy_bow_corpus = lazy_bow_corpus

    def __iter__(self):
        return self

    def __next__(self):
        return self.tfidf[next(self.lazy_bow_corpus)]


lda_model = gensim.models.LdaMulticore(LazyTfidfCorpus(tfidf, LazyBowCorpus(dictionary, LazyTokenCorpus(filenames))), num_topics=14, id2word=dictionary, passes=10, workers=4, alpha=0.01, eta=.91)


# does this mean I need to load each tweet 3 times!?
# remember to save the tf-idf model
