import csv
import json
import nltk
import numpy
from os import listdir, path
from pprint import pprint
import string


import label  # label.py for topic and sentiment ids


# Configuration.
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

# Language tools.
punctuation_set = set(string.punctuation)
stopwords_set = set(nltk.corpus.stopwords.words('english'))
stemmer = nltk.stem.snowball.SnowballStemmer('english')

class Index:
    def __init__(self, csv_type):
        # Machine learning data.
        self.feature_set = {}

        self.development_labels = self.read_csv(csv_type)
        self.tweets_data = self.read_tweets(directories.get('tweets'))
        self.tweet_features = self.generate_feature_vectors(self.tweets_data)

    def read_csv(self, csv_name):
        print('reading csv: ' + csv_name + '...')
        csv_labels = {}

        with open(files[csv_name]) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            next(csv_reader)  # Skip header row.
            for row in csv_reader:
                label_id = label.ids[row[0]][row[1]]
                csv_labels[row[2]] =  label_id

        print('done reading csv')
        return csv_labels

    def process_tweet(self, json_data):
        text = json_data.get('text')

        # Strip URLs.
        for url in json_data.get('entities').get('urls', []):
            text = text.replace(url.get('url', ''), '')

        # Strip words beginning with @ or #.
        # TODO: Strip "RT" at the beginning of the tweet?
        text_split = text.split(' ')
        text = ' '.join([x for x in text_split if len(x) > 0 and x[0] != '@' and x[0] != '#'])

        # Tokenize and remove punctuation and stopwords.
        # TODO: Might need to consider stopwords that tweak meanings of words, e.g. 'not'.
        tokens = nltk.word_tokenize(text)
        tokens = [x for x in tokens if x not in punctuation_set and x not in stopwords_set]

        # Stem the tokens.
        stemmed = [stemmer.stem(x) for x in tokens]

        # Add token to feature_set.
        # TODO: Maybe move this part out of this method.
        for x in stemmed:
            if x not in self.feature_set:
                self.feature_set[x] = len(self.feature_set)

        return stemmed

    def read_tweets(self, dir_name):
        print('reading tweets...')
        json_tweets = {}

        for f in listdir(dir_name):
            full_path = path.join(dir_name, f)

            tweet_id, f_ext = path.splitext(full_path)
            if (path.isfile(full_path) and f_ext == '.json'
                    and tweet_id[tweet_id.rfind('/')+1:] in self.development_labels):
                with open(full_path) as json_file:
                    json_data = json.load(json_file)
                    json_tweets[json_data['id']] = self.process_tweet(json_data)

        print('done reading tweets')
        return json_tweets

    def generate_feature_vectors(self, tweet_data):
        print('generating feature vectors...')
        tweet_features = {}

        for tweet_id, data in tweet_data.items():
            vector = numpy.zeros(len(self.feature_set))

            for token in data:
                if token in self.feature_set:
                    vector[self.feature_set[token]] += 1  # Not normalised.

            tweet_features[tweet_id] = vector

        print('done generating feature vectors')
        return tweet_features

