import nltk

from utility import punctuation_set, stemmer, stopwords_set

stopwords_set.add('RT')

def process_tweet(json_data):
    text = json_data.get('text')

    # Strip URLs.
    for url in json_data.get('entities').get('urls', []):
        text = text.replace(url.get('url', ''), '')

    # Tokenize and remove punctuation and stopwords.
    # TODO: Might need to consider stopwords that tweak meanings of words, e.g. 'not'.
    tokens = nltk.word_tokenize(text)
    tokens = [x for x in tokens if x not in punctuation_set and x not in stopwords_set]

    # Stem the tokens.
    stemmed = [stemmer.stem(x) for x in tokens]

    return stemmed
