import csv
import nltk
import string

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

def read_lexicon(csv_name):
    lexicon = {}

    with open(csv_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        for row in csv_reader:
            lexicon[row[0]] = row[1]

    return lexicon

# Language tools.
punctuation_set = set(string.punctuation)
stopwords_set = set(nltk.corpus.stopwords.words('english'))
stemmer = nltk.stem.snowball.SnowballStemmer('english')
lexicon = read_lexicon(paths['files']['lexicon'])
