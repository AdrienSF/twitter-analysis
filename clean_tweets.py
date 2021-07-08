from multiprocessing import Process
import pickle
from datetime import datetime
import os, json


def pickle_tweets(filenames: list):
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
        to_save = []
        for tweet in tweets:
            if 'full_text' in tweet:
                ttext = tweet['full_text']
            elif 'extended_tweet' in tweet and 'full_text' in tweet['extended_tweet']:
                ttext = tweet['extended_tweet']['full_text']
            else:
                ttext = tweet['text']

            date = tweet['created_at']

            to_save.append((date, ttext))

        all_tweets = all_tweets + to_save
    

    # get new filename
    start_date = all_tweets[0][0]
    start_date = datetime.strftime(datetime.strptime(start_date,'%a %b %d %H:%M:%S +0000 %Y'), '%Y-%m-%d %H:%M:%S')
    start_date = date.replace(' ', '_').replace(':','-')

    end_date = all_tweets[-1][0]
    end_date = datetime.strftime(datetime.strptime(end_date,'%a %b %d %H:%M:%S +0000 %Y'), '%Y-%m-%d %H:%M:%S')
    end_date = date.replace(' ', '_').replace(':','-')


    new_filename = start_date + '--' + end_date + '.pickle'
    

    with open('clean_data/'+new_filename, 'wb') as f:
        pickle.dump(all_tweets, f)


def split_job(filenames: list):
    # pickle 100 files at a time
    for i in range(0, len(filenames), 100):
        pickle_tweets(filenames[i:i + 100])



# use 8 processes to clean all the data
all_filenames = os.listdir('twitter_data')
all_filenames = [ 'twitter_data/'+filename for filename in all_filenames ]
eighth = int(len(all_filenames)/8)
procs = []
for i in range(8):
    filenames = all_filenames[i*eighth:(i+1)*eighth]
    procs.append(Process(target=split_job, args=(filenames,)))

# do the remainder first
split_job(all_filenames[8*eighth:])
print('finished remainder, starting 8 chunks')
# start the 8 processes
for proc in procs:
    proc.start()

# join the 8 processes
for proc in procs:
    proc.join()



