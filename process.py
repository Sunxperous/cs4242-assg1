import nltk

from utility import punctuation_set, stemmer, stopwords_set

def process_tweet(json_data):
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

    return stemmed
