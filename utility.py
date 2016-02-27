import csv

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


def read_lexicon(csv_name):
    lexicon = {}

    with open(csv_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        for row in csv_reader:
            # Strip # if there is.
            if not row[0].find('#'):
                word = row[0][1:]
            else:
                word = row[0]
            if len(word.split(' ')) == 1:
                lexicon[word] = row[1]

    return lexicon

# Language tools.
lexicon = read_lexicon(files['lexicon'])
