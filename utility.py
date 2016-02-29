import csv
import yaml

# Read default configuration.
with open('config.default.yml') as j:
    default_config = yaml.load(j.read())
files = default_config.get('files')
directories = default_config.get('directories')
constants = default_config.get('constants')
toggles = default_config.get('toggles')

# Read custom configuration.
try:
    with open('config.yml') as j:
        custom_config = yaml.load(j.read())
except FileNotFoundError:
    custom_config = {}
finally:
    if custom_config == None:
        custom_config = {}
files.update(custom_config.get('files', {}))
directories.update(custom_config.get('directories', {}))
constants.update(custom_config.get('constants', {}))
toggles.update(custom_config.get('toggles', {}))


# Lexicon.
def read_lexicon(csv_name):
    lexicon = {}

    with open(csv_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\n')
        for row in csv_reader:
            lexicon[row[0]] = row[0]

    return lexicon

positive = read_lexicon(files['positive'])
negative = read_lexicon(files['negative'])

# Labels.
label_ids = dict()
count = 0
for topic in ['apple', 'google', 'microsoft', 'twitter']:
    label_ids[topic] = dict()
    for sentiment in ['positive', 'negative', 'neutral', 'irrelevant']:
        label_ids[topic][sentiment] = count
        count += 1
# count should be 16.

