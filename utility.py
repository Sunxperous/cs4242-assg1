import nltk
import string


# Language tools.
punctuation_set = set(string.punctuation)
stopwords_set = set(nltk.corpus.stopwords.words('english'))
stemmer = nltk.stem.snowball.SnowballStemmer('english')

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

paths = {'files': files, 'directories': directories}

token_minimum_count = 1
