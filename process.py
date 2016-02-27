import nltk

from utility import punctuation_set, stemmer, stopwords_set

for w in ['don', 'no', 'not']:
    stopwords_set.remove(w)
stopwords_set.add('RT')


twitter_tokenizer = nltk.tokenize.TweetTokenizer(reduce_len=3)

def process_tweet(json_data):
    text = json_data.get('text')

    # Strip URLs.
    for url in json_data.get('entities').get('urls', []):
        text = text.replace(url.get('url', ''), 'http')

    # Tokenize and remove punctuation and stopwords.
    # TODO: Might need to consider stopwords that tweak meanings of words, e.g. 'not'.
    tokens = twitter_tokenizer.tokenize(text)
    tokens = [x for x in tokens if x not in punctuation_set and x not in stopwords_set]

    # Stem the tokens.
    stemmed = [stemmer.stem(x) for x in tokens]

    result = {}
    result['stemmed'] = stemmed
    result['user'] = json_data.get('user')

    return result
