import csv
import json
import nltk
from numpy import zeros
from os import listdir, path
from pprint import pprint


import label  # label.py for topic and sentiment ids


# Set up.
from configparser import SafeConfigParser
config = SafeConfigParser()
# Read default configuration.
config.read('config.default.ini')
files = dict(config.items('file'))
directories = dict(config.items('directory'))
# Read custom configuration.
config.read('config.ini')
files.update(config.items('file'))
directories.update(config.items('directory'))

stopwords_set = set(nltk.corpus.stopwords.words('english'))


def read_csv(csv_name):
    print('reading csv: ' + csv_name)
    csv_labels = dict()

    with open(files[csv_name]) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader)  # Skip header row.
        for row in csv_reader:
            csv_labels[row[2]] = zeros(label.count)
            label_id = label.ids[row[0]][row[1]]
            csv_labels[row[2]][label_id] = 1

    print('done reading csv')
    return csv_labels


def process_tweet(json_data):
    text = json_data.get('text')

    # Strip URLs.
    for url in json_data.get('entities').get('urls', []):
        text = text.replace(url.get('url', ''), '')

    # Strip words beginning with @ or #.
    text = text.split(' ')
    text = ' '.join([x for x in text if x[0] != '@' and x[0] != '#'])

    words = nltk.word_tokenize(text)
    # TODO: Remove stopwords.
    # TODO: Use Snowball stemmer (possibly other languages).
    return words


def read_tweets(dir_name):
    print('reading tweets')
    json_tweets = dict()

    for f in listdir(dir_name):
        full_path = path.join(dir_name, f)

        if path.isfile(full_path) and path.splitext(full_path)[1] == '.json':
            with open(full_path) as json_file:
                json_data = json.load(json_file)
                json_tweets[json_data['id']] = process_tweet(json_data)
                return json_tweets

    print('done reading tweets')
    return json_tweets


development_labels = read_csv('development')
tweets_data = read_tweets(directories.get('tweets'))
pprint(tweets_data[next(iter(tweets_data))])
