import py7zr
import os

filenames = os.listdir('twitter_data')

# load some sample data
for filename in filenames:
    if '.7z' in filename:
        with py7zr.SevenZipFile('twitter_data/'+filename, 'r') as z:
            z.extractall(path='twitter_data')
        os.remove('twitter_data/'+filename)