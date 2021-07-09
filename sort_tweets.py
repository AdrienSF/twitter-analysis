# how to sort something too large to fit in memory?

# sort every file individually, rename the files to correct bounds


# never mind, I think it can all fit into mem?... nope

from multiprocessing import Process
import pickle
from datetime import datetime, timedelta
import os, json, gc
from guppy import hpy; h=hpy()

from helpers import log



def sort_tweets(filenames: list):
    all_tweets = []
    # load tweets
    for i in range(len(filenames)):
        filename = filenames[i]
        with open(filename, 'rb') as f:
            all_tweets = all_tweets + pickle.load(f)
        gc.collect()
        log('loaded files: '+ str(i))
        log('mem:'+ str(h.heap().size))


    log('total tweets:'+ str(len(all_tweets)))

    # change each date str to a date object
    log('changing tweet date str to datetime obj...')
    all_tweets = [ (datetime.strptime(tweet[0],'%a %b %d %H:%M:%S +0000 %Y'), tweet[1]) for tweet in all_tweets ]
    gc.collect()
    log('done')

    # sort tweets by date(first of tuple)
    log('sorting...')
    # all_tweets = sorted(all_tweets, key=lambda tweet: tweet[0])
    # sort in place
    all_tweets.sort(key=lambda tweet: tweet[0])
    gc.collect()
    log('done')

    log('saving chunks...')
    while all_tweets:
        # split tweets into week long chunks
        chunk = [all_tweets.pop(0)]
        while chunk[-1][0] < chunk[0][0] + timedelta(days=7) and all_tweets:
            chunk.append(all_tweets.pop(0))

        # get new filename
        start_date = chunk[0][0]
        start_date = datetime.strftime(start_date, '%Y-%m-%d_%H-%M-%S')

        end_date = chunk[-1][0]
        end_date = datetime.strftime(end_date, '%Y-%m-%d_%H-%M-%S')


        new_filename = start_date + '--' + end_date + '.pickle'
        
        # write file
        with open('sorted_data/'+new_filename, 'wb') as f:
            pickle.dump(chunk, f)





all_filenames = os.listdir('clean_data')
all_filenames = [ 'clean_data/'+filename for filename in all_filenames ]
# let's hope this ends up in order...
sort_tweets(all_filenames[:125])
sort_tweets(all_filenames[125:250])
sort_tweets(all_filenames[250:])

