from collections import defaultdict, OrderedDict
import csv
import json
import numpy
from os import listdir, path
from pprint import pprint

from process import process_tweet
import label  # label.py for topic and sentiment ids
from utility import lexicon, paths, token_minimum_count


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
        self.tweet_labels = self.read_labels(csv_type)
        print('labelled ' + str(len(self.tweet_labels)) + ' tweets')
        print('reading tweets...')
        self.tweet_data = self.read_tweets(paths['directories'].get('tweets'))
        print('read ' + str(len(self.tweet_data)) + ' tweets')

        print('adding into feature set...')
        self.add_to_feature_set(self.tweet_data)
        print('added ' + str(len(self.feature_set)) + ' (word) features')

        print('generating feature vectors...')
        self.tweet_features = self.generate_feature_vectors(self.tweet_data)
        print('generated ' + str(len(self.tweet_features)) + ' feature vectors')
        print('indexing complete!\n')

    def read_labels(self, csv_name):
        tweet_labels = OrderedDict()

        with open(paths['files'][csv_name]) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            next(csv_reader)  # Skip header row.
            for row in csv_reader:
                label_id = label.ids[row[0]][row[1]]
                tweet_labels[row[2]] =  label_id

        return tweet_labels

    def add_to_feature_set(self, stemmed_tweets):
        """Add token to feature_set if it appears more than twice."""
        if not self.train:
            return

        word_tokens = defaultdict(int)

        for tweet_id, stemmed_words in stemmed_tweets.items():
            for word in stemmed_words:
                word_tokens[word] += 1

        i = 0
        for word, count in word_tokens.items():
            if count >= token_minimum_count:
                self.feature_set[word] = i
                i += 1

    def add_lexicon(self, lexicon):
        pass

    def read_tweets(self, dir_name):
        json_tweets = OrderedDict()

        # TODO: Iterate on tweet_labels list instead of iterating on directory.
        for f in listdir(dir_name):
            full_path = path.join(dir_name, f)

            tweet_id, f_ext = path.splitext(full_path)
            if (path.isfile(full_path) and f_ext == '.json'
                    and tweet_id[tweet_id.rfind('/')+1:] in self.tweet_labels):
                with open(full_path) as json_file:
                    json_data = json.load(json_file)
                    json_tweets[json_data['id']] = process_tweet(json_data)

        return json_tweets

    def generate_feature_vectors(self, tweet_data):
        tweet_features = OrderedDict()

        for tweet_id, data in tweet_data.items():
            vector = numpy.zeros(len(self.feature_set))

            for token in data:
                if token in self.feature_set:
                    vector[self.feature_set[token]] += 1  # Not normalised.

            tweet_features[tweet_id] = vector

        return tweet_features

