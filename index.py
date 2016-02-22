import csv
import json
import nltk
import numpy
from os import listdir, path
from pprint import pprint


import label  # label.py for topic and sentiment ids
from utility import paths, punctuation_set, stemmer, stopwords_set


class Index:
    """
    Attributes:
        feature_set: All the terms used to generate the feature vectors.
        train: Whether to add terms to the feature_set or not.
        tweet_labels: Actual topic and sentiment labels for given tweet.
        tweet_data: Raw tweet data.
        tweet_features: Tweet data transformed into feature vectors.
    """
    def __init__(self, csv_type, feature_set=None):
        """
        Args:
            csv_type: Either 'development', 'training', or 'testing'.
            feature_set: Whether to use a fixed feature set, or None.
        """

        if feature_set is None:
            self.feature_set = {}
            self.train = True
            print('indexing with training...')
        else:
            self.feature_set = feature_set
            self.train = False
            print('indexing without training...')

        print('reading csv ' + csv_type + '...')
        self.tweet_labels = self.read_csv(csv_type)
        print('labelled ' + str(len(self.tweet_labels)) + ' tweets')
        print('reading tweets...')
        self.tweet_data = self.read_tweets(paths['directories'].get('tweets'))
        print('read ' + str(len(self.tweet_data)) + ' tweets')
        print('generating feature vectors...')
        self.tweet_features = self.generate_feature_vectors(self.tweet_data)
        print('generated ' + str(len(self.tweet_features)) + ' feature vectors')
        print('indexing complete.')

    def read_csv(self, csv_name):
        tweet_labels = {}

        with open(paths['files'][csv_name]) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            next(csv_reader)  # Skip header row.
            for row in csv_reader:
                label_id = label.ids[row[0]][row[1]]
                tweet_labels[row[2]] =  label_id

        return tweet_labels

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
            if x not in self.feature_set and self.train:
                self.feature_set[x] = len(self.feature_set)

        return stemmed

    def read_tweets(self, dir_name):
        json_tweets = {}

        # TODO: Iterate on tweet_labels list instead of iterating on directory.
        for f in listdir(dir_name):
            full_path = path.join(dir_name, f)

            tweet_id, f_ext = path.splitext(full_path)
            if (path.isfile(full_path) and f_ext == '.json'
                    and tweet_id[tweet_id.rfind('/')+1:] in self.tweet_labels):
                with open(full_path) as json_file:
                    json_data = json.load(json_file)
                    json_tweets[json_data['id']] = self.process_tweet(json_data)

        return json_tweets

    def generate_feature_vectors(self, tweet_data):
        tweet_features = {}

        for tweet_id, data in tweet_data.items():
            vector = numpy.zeros(len(self.feature_set))

            for token in data:
                if token in self.feature_set:
                    vector[self.feature_set[token]] += 1  # Not normalised.

            tweet_features[tweet_id] = vector

        return tweet_features

