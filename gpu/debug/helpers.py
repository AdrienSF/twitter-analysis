import json, random
import os, sys
import re
import gensim
from gensim.models import CoherenceModel
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import nltk
import numpy as np
import numpy.random

import tensorly as tl

# nltk.download('wordnet')

def loss_rec(factor, cumulant, theta):
    rec = tl.cp_to_tensor((None, [factor]*3))
    #rec_loss = tl.tenalg.inner(rec, cumulant)
    rec_loss = tl.norm(rec - cumulant, 2)**2
    ortho_loss = (1 + theta)/2*tl.norm(rec, 2)**2
    return ortho_loss + rec_loss, ortho_loss, rec_loss/tl.norm(cumulant, 2)

def loss_15(factor, cumulant, theta):
    rec = tl.cp_to_tensor((None, [factor]*3))
    rec_loss = tl.norm(rec, 2)**2 + tl.norm(cumulant, 2)**2 - 2*tl.tenalg.inner(rec, cumulant) 
    # rec_loss = tl.norm(rec - cumulant, 2)**2
    ortho_loss = (1 + theta)/2*tl.norm(rec, 2)**2
    return ortho_loss + rec_loss, ortho_loss, rec_loss/tl.norm(cumulant, 2)


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
    ttext = stemmer.stem(lemmatizer.lemmatize(text, pos='v'))
    return re.sub(r"http\S+", "http", ttext)

def get_preprocessed(text, stemmer, lemmatizer):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3: #[NOTE]: maybe should relax this restriction of >3
            result.append(sys.intern(lemmatize_stemming(token, stemmer, lemmatizer)))
    return result

gtpjoin = lambda text: ' '.join(get_preprocessed(text, SnowballStemmer('english'), WordNetLemmatizer()))
gtp = lambda text: get_preprocessed(text, SnowballStemmer('english'), WordNetLemmatizer())

def load_tweets(filenames, preprocessor=gtp, subsample_proportion=1):

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
                ttext = tweet['full_text']
            elif 'extended_tweet' in tweet and 'full_text' in tweet['extended_tweet']:
                ttext = tweet['extended_tweet']['full_text']
            else:
                ttext = tweet['text']

            # remove/replace URLs
            # ttext = re.sub(r"http\S+", "http", ttext) # too slow

            if preprocessor:
                ttext = preprocessor(ttext)
            ttexts.append(ttext)

        all_tweets = all_tweets + random.sample(ttexts, int(len(ttexts)*subsample_proportion))


    return all_tweets











def get_phi(top_n, vocab_size, doc_num, t_per_doc,alpha_val=1.5):
    np.random.seed(seed=0)
    '''use code here:
    http://www.hongliangjie.com/2010/09/30/generate-synthetic-data-for-lda/
    to get document, topic matrices'''
    ## define some constant
    TOPIC_N = top_n
    VOCABULARY_SIZE = vocab_size
    DOC_NUM = doc_num
    TERM_PER_DOC = t_per_doc
    w_arr = np.zeros((DOC_NUM, VOCABULARY_SIZE), dtype=np.float32)

    #beta = [0.01 for i in range(VOCABULARY_SIZE)]
    #alpha = [0.9 for i in range(TOPIC_N)]
    beta = [0.01 for i in range(VOCABULARY_SIZE)] # 0.5
    alpha = [alpha_val for i in range(TOPIC_N)]

    phi = []
    theta_arr = np.zeros((DOC_NUM, TOPIC_N))
    ## generate multinomial distribution over words for each topic
    for i in range(TOPIC_N):
    	topic =	numpy.random.mtrand.dirichlet(beta, size = 1)
    	phi.append(topic)

    for i in range(DOC_NUM):
    	buffer = {}
    	z_buffer = {} ## keep track the true z
    	## first sample theta
    	theta = numpy.random.mtrand.dirichlet(alpha,size = 1)
    	for j in range(TERM_PER_DOC):
    		## first sample z
    		z = numpy.random.multinomial(1,theta[0],size = 1)
    		z_assignment = 0
    		for k in range(TOPIC_N):
    			if z[0][k] == 1:
    				break
    			z_assignment += 1
    		if not z_assignment in z_buffer:
    			z_buffer[z_assignment] = 0
    		z_buffer[z_assignment] = z_buffer[z_assignment] + 1
    		## sample a word from topic z
    		w = numpy.random.multinomial(1,phi[z_assignment][0],size = 1)
    		w_assignment = 0
    		for k in range(VOCABULARY_SIZE):
    			if w[0][k] == 1:
    				break
    			w_assignment += 1
    		if not w_assignment in buffer:
    			buffer[w_assignment] = 0
    		buffer[w_assignment] = buffer[w_assignment] + 1
    		w_arr[i] = w_arr[i] + w
    	theta_arr[i] = theta
    return tl.tensor(w_arr), phi, theta_arr, sum(alpha)