import os
import json
import random
import sys
from datetime import date
import gensim
from helpers import lemmatize_stemming, get_preprocessed, compute_coherence_values, load_tweets
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import nltk
nltk.download('wordnet')

filenames = os.listdir('twitter_data')
filenames = [ 'twitter_data/'+filename for filename in filenames ]


save_dirname = 'trained-' + str(date.today())
# redo everything  but with yield, and faster?


class TweetLoader:

    """Iterable that loades, filters, stems and vectorizes tweets (english non-retweets only)"""

    def __init__(self, filenames):
        self.filenames = filenames

        stemmer = SnowballStemmer('english')
        lemmatizer = WordNetLemmatizer()
        self.preprocess = lambda text: get_preprocessed(text, stemmer, lemmatizer)

    def __iter__(self):
        for i in range(len(self.filenames)):
            filename = self.filenames[i]
            if i % int(len(self.filenames)/10) == 0:
                print('loading files:', int(100*i/len(self.filenames)), '%')
            with open(filename, 'r') as f:
                    # add commas between tweets to correct json syntax
                tweet_list = json.loads('['+f.read().replace('}{','},{')+']')

            for tweet in tweet_list:
                if 'retweeted_status' not in tweet and tweet['lang'] == 'en':
                    if 'full_text' in tweet:
                        text = tweet['full_text']
                    elif 'extended_tweet' in tweet and 'full_text' in tweet['extended_tweet']:
                        text = tweet['extended_tweet']['full_text']
                    else:
                        text = tweet['text']


                    yield self.preprocess(text)



tweet_loader = TweetLoader(filenames)
print('building dictionary...')
dictionary = gensim.corpora.Dictionary(tweet_loader)
if not os.path.isdir(save_dirname):
    os.makedirs(save_dirname)
dictionary.save(save_dirname + '/dictionary')
print('saved')


class BowCorpus:
    def __init__(self, dictionary, token_corpus):
        self.token_corpus = token_corpus
        self.dictionary = dictionary

    def __iter__(self):
        for tokenized in self.token_corpus:
            yield self.dictionary.doc2bow(tokenized)

print('building tfidf')
tfidf = gensim.models.TfidfModel(BowCorpus(dictionary, tweet_loader))
tfidf.save(save_dirname + '/tfidf')
print('saved')


class TfidfCorpus:
    def __init__(self, tfidf_model, BowCorpus):
        self.tfidf = tfidf_model
        self.bow_corpus = BowCorpus

    def __iter__(self):
        for doc in self.bow_corpus:
            yield self.tfidf[doc]

print('building lda model')
lda_model = gensim.models.LdaMulticore(TfidfCorpus(tfidf, BowCorpus(dictionary, tweet_loader)), num_topics=14, id2word=dictionary, passes=5, workers=8, alpha=0.01, eta=.91)
lda_model.save(save_dirname + '/trained_lda')
print('saved')

# for idx, topic in lda_model.print_topics(-1):
#     print('Topic: {} \nWords: {}'.format(idx, topic))