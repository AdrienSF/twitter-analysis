import json
import os
import gensim
from gensim.models import CoherenceModel
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import nltk
# nltk.download('wordnet')



def compute_coherence_values(corpus, dictionary, k, a='symmetric', b=None, coherence='u_mass', texts=None):
    
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=dictionary,
                                           num_topics=k, 
                                           random_state=100,
                                           chunksize=100,
                                           passes=10,
                                           alpha=a,
                                           eta=b,
                                           workers=10)
    
    if coherence == 'u_mass':
        coherence_model_lda = CoherenceModel(model=lda_model, corpus=corpus, coherence=coherence, processes=4)
    else:
        coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, coherence=coherence, processes=4)

    # print('coherence: ', coherence_model_lda.get_coherence())
    
    return coherence_model_lda.get_coherence()




def lemmatize_stemming(text, stemmer, lemmatizer):
    # return SnowballStemmer('english').stem(WordNetLemmatizer().lemmatize(text, pos='v'))
    return stemmer.stem(lemmatizer.lemmatize(text, pos='v'))

def get_preprocessed(text, stemmer, lemmatizer):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3: #[NOTE]: maybe should relax this restriction of >3
            result.append(lemmatize_stemming(token, stemmer, lemmatizer))
    return result


def load_tweets(filenames):

    all_tweets = []
    for filename in filenames:
        with open(filename, 'r') as f:
            # add commas between tweets to correct json syntax
            data = json.loads('['+f.read().replace('}{','},{')+']')
        # remove retweets
        tweets = [tweet for tweet in data if 'retweeted_status' not in tweet]
        # keep english language tweets only
        tweets = [tweet for tweet in tweets if tweet['lang'] == 'en']

        # take tweet text  or full_text if the tweet has that attribute
        ttexts = []
        for tweet in tweets:
            if 'full_text' in tweet:
                ttexts.append(tweet['full_text'])
            elif 'extended_tweet' in tweet and 'full_text' in tweet['extended_tweet']:
                ttexts.append(tweet['extended_tweet']['full_text'])
            else:
                ttexts.append(tweet['text'])

        all_tweets = all_tweets + tweets


    return all_tweets