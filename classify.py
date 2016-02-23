from sklearn import neighbors


from index import Index


trained_index = Index('training')

knn = neighbors.KNeighborsClassifier(weights='distance')
samples = [v for k, v in trained_index.tweet_features.items()]
targets = [v for k, v in trained_index.tweet_labels.items()]

knn.fit(samples, targets)

testing_index = Index('testing', trained_index.feature_set)

prediction = {}
for tweet_id, feature_vector in testing_index.tweet_features.items():
    prediction[tweet_id] = knn.predict([feature_vector])[0]


#print([k for k, v in testing_index.tweet_features.items()])
#print([k for k, v in prediction.items()])
#exit()

correct = 0
wrong = 0
for tweet_id, predicted in prediction.items():
    if str(tweet_id) not in testing_index.tweet_labels:
        print(str(tweet_id) + ' not found in tweet_labels, wtf?')
    else:
        actual = testing_index.tweet_labels[str(tweet_id)]
        if predicted % 4 != actual % 4:  # Check only for sentiments ignoring topic.
            wrong += 1
            print('predicted ' + str(predicted) + ' but actually is ' + str(actual))
        else:
            correct += 1

print('correct: ' + str(correct) + '; wrong: ' + str(wrong))


