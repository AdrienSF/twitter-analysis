import json, os
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
save_dirname = 'trained-' + str(date.today())
if not os.path.isdir(save_dirname):
    os.makedirs(save_dirname)


filenames = os.listdir('data/unzipped')
filenames = [[ 'data/unzipped/'+filename for filename in filenames ][1]]

tweets = load_tweets(filenames)

log('loading saved dictionary...')
dictionary = Dictionary.load('trained-2021-06-02/filtered_dictionary')
log('loaded')

log('loading saved tfidf...')
tfidf = TfidfModel.load('trained-2021-06-09/tfidf')
log('loaded')


bow_corpus = [dictionary.doc2bow(tweet) for tweet in tweets]
tfidf_corpus = tfidf[bow_corpus]

log('building lda model')
lda_model = LdaMulticore(tfidf_corpus, num_topics=14, id2word=dictionary, passes=5, workers=8)
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