import json
import os
import gensim
from gensim.models import CoherenceModel
from nltk.stem import WordNetLemmatizer, SnowballStemmer


def load_tweets(filenames, full_text_only=True):
    all_tweets = []
    for filename in filenames:
        # print(filename)
        with open(filename, 'r') as f:
            data = json.loads('['+f.read().replace('}{','},{')+']')
        # remove retweets
        tweets = [tweet for tweet in data if 'retweeted_status' not in tweet]
        # keep english language tweets only
        tweets = [tweet for tweet in tweets if tweet['lang'] == 'en']

        # print(tweets[1])

        # seperate out just tweet text for simplicity
        # for tweet in tweets:
        #     print('full_text' in tweet)
        if full_text_only:
            ttexts = [ tweet['full_text'] for tweet in tweets if 'full_text' in tweet ]
        else:
            ttexts = [ tweet['full_text'] if 'full_text' in tweet else tweet['text'] for tweet in tweets]

        # (probably unecessary) keep an id map: tweet text index --> tweet id
        # id_map = {i: tweets[i]['id'] for i in range(len(tweets))}
        # for i in range(len(ttexts)):
        #     print(('full_text' in tweets[i]), ttexts[i])

        # remove numbers
        # ttexts = [re.sub('[0-9]+', '', tweet) for tweet in ttexts]
        all_tweets = all_tweets + ttexts


    return all_tweets



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




def lemmatize_stemming(text):
    return SnowballStemmer('english').stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result