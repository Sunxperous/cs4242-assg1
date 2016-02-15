ids = dict()

count = 0
for topic in ['apple', 'google', 'microsoft', 'twitter']:
    ids[topic] = dict()
    for sentiment in ['positive', 'negative', 'neutral', 'irrelevant']:
        ids[topic][sentiment] = count
        count += 1

# count should be 16.

