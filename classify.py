from sklearn import neighbors
from sklearn import svm

from index import Index


trained_index = Index('training')

samples = [v for k, v in trained_index.tweet_features.items()]
targets = [v for k, v in trained_index.tweet_labels.items()]

knn = neighbors.KNeighborsClassifier(10, weights='uniform')
knn.fit(samples, targets)

svc = svm.SVC(kernel='linear', C=1.5)
# linear kernel is used when feature size is big (~10,000) and sample size is moderate (5000)
# C is default 1.0, decrease if overfitting, increase if underfitting
# if uneven data result, try adding (class_weight='balanced')
# scaling and normalization is highly recommended
#svc.fit(samples, targets)

testing_index = Index('testing', trained_index.feature_set)

classifier = knn
#classifier = svc
prediction = {}
for tweet_id, feature_vector in testing_index.tweet_features.items():
    prediction[tweet_id] = classifier.predict([feature_vector])[0]


#print([k for k, v in testing_index.tweet_features.items()])
#print([k for k, v in prediction.items()])
#exit()

correct = 0
wrong = 0

sentiments = ['positive', 'negative', 'neutral', 'irrelevant']
results = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

for tweet_id, predicted in prediction.items():
    if str(tweet_id) not in testing_index.tweet_labels:
        print(str(tweet_id) + ' not found in tweet_labels, wtf?')
    else:
        actual = testing_index.tweet_labels[str(tweet_id)]
        results[predicted % 4][actual % 4] += 1
        if predicted % 4 != actual % 4:  # Check only for sentiments ignoring topic.
            wrong += 1
            print(tweet_id)
            # print('predicted ' + str(predicted) + ' but actually is ' + str(actual))
            print('predicted ' + sentiments[predicted % 4] + ' but actually is ' + sentiments[actual % 4])

        else:
            correct += 1

print('wrong result: ' + str(results))

print('correct: ' + str(correct) + '; wrong: ' + str(wrong))


