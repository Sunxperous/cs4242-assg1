from collections import defaultdict, OrderedDict
import csv
import json
import numpy
from os import listdir, path
from pprint import pprint
import random

from process import process_tweet
import label  # label.py for topic and sentiment ids
from utility import positive, negative, paths, token_minimum_count


class Index:
    """
    Attributes:
        feature_set: All the terms used to generate the feature vectors.
        train: Whether to add terms to the feature_set or not.
        tweet_labels: Actual topic and sentiment labels for given tweet.
        tweet_data: Processed raw tweet data.
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
        self.add_lexicon_to_feature_set(positive)
        self.add_lexicon_to_feature_set(negative)
        print('added ' + str(len(self.feature_set)) + ' (word) features')

        print('generating feature vectors...')
        self.tweet_features = self.generate_feature_vectors(self.tweet_data)
        print('generated up to ' + str(len(self.tweet_features)) + ' feature vectors')
        print('creating feature vectors from lexicon...')

        self.generate_lexicon_data(positive, True)
        self.generate_lexicon_data(negative, False)
        
        print('generated up to ' + str(len(self.tweet_features)) + ' feature vectors')
        print('indexing complete!\n')

    def read_labels(self, csv_name):
        tweet_labels = OrderedDict()

        with open(paths['files'][csv_name]) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            next(csv_reader)  # Skip header row.
            for row in csv_reader:
                label_id = label.ids[row[0]][row[1]] % 4  # Use 4 labels instead of 16.
                tweet_labels[int(row[2])] =  label_id

        return tweet_labels

    def add_to_feature_set(self, stemmed_tweets):
        """Add token to feature_set if it appears more than twice."""
        if not self.train:
            return

        word_tokens = defaultdict(int)

        for tweet_id, data in stemmed_tweets.items():
            for word in data['stemmed']:
                word_tokens[word] += 1

        i = 0
        for word, count in word_tokens.items():
            if count >= token_minimum_count:
                self.feature_set[word] = i
                i += 1

    def add_lexicon_to_feature_set(self, lexicon):
        i = len(self.feature_set)
        for word, v in lexicon.items():
            if word not in self.feature_set:
                self.feature_set[word] = i
                i += 1

    def read_tweets(self, dir_name):
        json_tweets = OrderedDict()

        # TODO: Iterate on tweet_labels list instead of iterating on directory.
        for f in listdir(dir_name):
            full_path = path.join(dir_name, f)

            tweet_id, f_ext = path.splitext(full_path)
            if (path.isfile(full_path) and f_ext == '.json'
                    and int(tweet_id[tweet_id.rfind('/')+1:]) in self.tweet_labels):
                with open(full_path) as json_file:
                    json_data = json.load(json_file)
                    json_tweets[json_data['id']] = process_tweet(json_data)

        return json_tweets

    def generate_feature_vectors(self, tweet_data):
        tweet_features = OrderedDict()

        for tweet_id, data in tweet_data.items():
            vector = numpy.zeros(len(self.feature_set))

            # Token level features.
            for token in data['stemmed']:
                if token in self.feature_set:
                    vector[self.feature_set[token]] += 1  # Not normalised.

            # Social features.
            #user_data = data['user']
            #vector = numpy.append(vector, user_data['followers_count'])
            #vector = numpy.append(vector, user_data['friends_count'])
            #vector = numpy.append(vector, user_data['listed_count'])
            #vector = numpy.append(vector, user_data['statuses_count'])
            tweet_features[tweet_id] = vector

        return tweet_features

    def generate_lexicon_data(self, lexicon, isPositive):
        if not self.train:
            return

        lexicon_vectors = OrderedDict()
        lexicon_labels = OrderedDict()

        key = random.getrandbits(32)
        for word, v in lexicon.items():
            if word in self.feature_set:
                continue

            """
            if float(weight) >= 0.15 and float(weight) <= 0.5:
                continue
            """

            # Generate feature vector of word.
            lexicon_vectors[key] = numpy.zeros(len(self.feature_set))
            lexicon_vectors[key][self.feature_set[word]] += 1

            # Generate label of word.
            if isPositive:
                lexicon_labels[key] = 0
            else:
                lexicon_labels[key] = 1

            key += 1

        self.tweet_features.update(lexicon_vectors)
        self.tweet_labels.update(lexicon_labels)


