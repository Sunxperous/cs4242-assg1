from sklearn import neighbors
from sklearn import svm

from index import Index
from utility import constants, toggles

model = int(input('Input 1 for KNN and 2 for SVM '))

if model == 1:
    num_neighbours = int(input('Input Number of Neighbours for KNN '))
elif model == 2:
    penalty_constant = int(input('Input Penalty Constant, C '))
else:
    exit()

trained_index = Index(constants['file_to_train'])

samples = [v for k, v in sorted(trained_index.tweet_features.items(), key=lambda t: t[0])]
targets = [v for k, v in sorted(trained_index.tweet_labels.items(), key=lambda t: t[0])]

if model == 1:
    clf = neighbors.KNeighborsClassifier(n_neighbors=num_neighbours, weights='uniform')
    # n_neighbors is default 5
elif model == 2:
    clf = svm.SVC(kernel='linear', C=penalty_constant)
    # linear kernel is used when feature size is big (~10,000) and sample size is moderate (5000)
    # C is default 1.0, decrease if overfitting, increase if underfitting
    # if uneven data result, try adding (class_weight='balanced')
    # scaling and normalization is highly recommended

clf.fit(samples, targets)

testing_index = Index(constants['file_to_test'], trained_index.feature_set)

prediction = {}
for tweet_id, feature_vector in testing_index.tweet_features.items():
    prediction[tweet_id] = clf.predict([feature_vector])[0]

correct = 0
wrong = 0

sentiments = ['positive', 'negative', 'neutral', 'irrelevant']
results = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

for tweet_id, predicted in prediction.items():
    if tweet_id not in testing_index.tweet_labels:
        print(tweet_id + ' not found in tweet_labels, wtf?')
    else:
        actual = testing_index.tweet_labels[tweet_id]
        results[actual % 4][predicted % 4] += 1
        if predicted % 4 != actual % 4:  # Check only for sentiments ignoring topic.
            wrong += 1
        else:
            correct += 1

positive_count = 0
negative_count = 0
neutral_count = 0
irrelevant_count = 0

for result in results[0]:
    positive_count += result

for result in results[1]:
    negative_count += result

for result in results[2]:
    neutral_count += result

for result in results[3]:
    irrelevant_count += result

print('* Out of ' + str(positive_count) + ' positive tweets...',)
for i, result in enumerate(results[0]):
    print('classified ' + sentiments[i] + ': ' + str(result) + ' (' + str(format(result/positive_count * 100, '.2f')) + '%)')

print('* Out of ' + str(negative_count) + ' negative tweets...',)
for i, result in enumerate(results[1]):
    print('classified ' + sentiments[i] + ': ' + str(result) + ' (' + str(format(result/negative_count * 100, '.2f')) + '%)')

print('* Out of ' + str(neutral_count) + ' neutral tweets...',)
for i, result in enumerate(results[2]):
    print('classified ' + sentiments[i] + ': ' + str(result) + ' (' + str(format(result/neutral_count * 100, '.2f')) + '%)')

print('* Out of ' + str(irrelevant_count) + ' irrelevant tweets...',)
for i, result in enumerate(results[3]):
    print('classified ' + sentiments[i] + ': ' + str(result) + ' (' + str(format(result/irrelevant_count * 100, '.2f')) + '%)')

print('*** Out of ' + str(correct + wrong) + ' total tweets...')
print('correct: ' + str(correct) + '; wrong: ' + str(wrong) + ' (' + str(format(correct/(correct+wrong) * 100, '.2f')) + '%)')


