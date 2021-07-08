# how to sort something too large to fit in memory?

# sort every file individually, rename the files to correct bounds


# never mind, I think it can all fit into mem?

from multiprocessing import Process
import pickle
from datetime import datetime
import os, json, gc


def sort_tweets(filenames: list):
    all_tweets = []
    # load tweets
    for filename in filenames:
        with open(filename, 'rb') as f:
            all_tweets = all_tweets + pickle.load(f)
    print('total tweet:', len(all_tweets))

    # change each date str to a date object
    print('changing tweet date str to datetime obj...')
    all_tweets = [ (datetime.strptime(tweet[0],'%a %b %d %H:%M:%S +0000 %Y'), tweet[1]) for tweet in all_tweets ]
    gc.collect()
    print(done)

    # sort tweets by date(first of tuple)
    print('sorting...')
    all_tweets = sorted(all_tweets, key=lambda tweet: tweet[0])
    gc.collect()
    print('done')

    print('saving chunks...')
    while all_tweets:
        # split tweets into week long chunks
        chunk = [all_tweets.pop(0)]
        while chunk[-1][0] < chunk[0][0] + datetime.timedelta(days=7) and all_tweets:
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




# use 8 processes to clean all the data
all_filenames = os.listdir('clean_data')
all_filenames = [ 'clean_data/'+filename for filename in all_filenames ]

sort_tweets(all_filenames)

