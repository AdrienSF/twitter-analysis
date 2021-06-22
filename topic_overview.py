import json, os, gc
from datetime import date
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from gensim.corpora import Dictionary
from gensim.models import TfidfModel, LdaMulticore, CoherenceModel
from helpers import load_tweets, compute_coherence_values

def log(message: str):
    message = str(message)
    with open('progress.log', 'a') as f:
        f.write(message+'\n')


dates = {
    'start':
    ['2020-04-23',
    '2020-04-24',
    '2020-04-25',
    '2020-04-26',
    '2020-04-27',
    '2020-04-28',
    '2020-04-29']
    ,
    'mid': 
    ['2020-06-12',
    '2020-06-13',
    '2020-06-14',
    '2020-06-15',
    '2020-06-16',
    '2020-06-17',
    '2020-06-18']
    ,
    'end':
    ['2020-08-11',
    '2020-08-12',
    '2020-08-13',
    '2020-08-14',
    '2020-08-15',
    '2020-08-16',
    '2020-08-17']
}

all_filenames = os.listdir('twitter_data')
all_filenames = [ 'twitter_data/'+filename for filename in all_filenames ]

log('loading saved dictionary...')
dictionary = Dictionary.load('trained-2021-06-02/filtered_dictionary')
log('loaded')

log('loading saved tfidf...')
tfidf = TfidfModel.load('trained-2021-06-09/tfidf')
log('loaded')
tweets = []
for week in dates:
    del tweets
    gc.collect()

    week_filenames = [ name for name in all_filenames if any(substring in name for substring in dates[week]) ]

    save_dirname = week + '-' + str(date.today())
    if not os.path.isdir(save_dirname):
        os.makedirs(save_dirname)

    log('loading tweets...')
    tweets = load_tweets(week_filenames, subsample_proportion=.05)


    log('builing bow corpus')
    bow_corpus = [dictionary.doc2bow(tweet) for tweet in tweets]
    tfidf_corpus = tfidf[bow_corpus]
    log('built')

    log('building lda model')
    lda_model = LdaMulticore(tfidf_corpus, num_topics=500, id2word=dictionary, passes=5, workers=8)
    lda_model.save(save_dirname + '/trained_lda')
    log('saved')



    # check coherence
    coherence_model_lda = CoherenceModel(model=lda_model, texts=tweets, coherence='c_v', processes=4)

    log(coherence_model_lda.get_coherence())


    # generate word cloud
    for t in range(lda_model.num_topics):
        plt.figure()
        plt.imshow(WordCloud(background_color='white').fit_words(dict(lda_model.show_topic(t, 200))))
        plt.axis("off")
        plt.title("Topic #" + str(t))
        plt.savefig(save_dirname+'/topic'+str(t)+'wordcloud.pdf', format='pdf')
