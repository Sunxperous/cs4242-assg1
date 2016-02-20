from sklearn import neighbors


from index import Index


index = Index('testing')

knn = neighbors.KNeighborsClassifier(weights='uniform')
sample = [v for k, v in index.tweet_features.items()]
target = [v for k, v in index.development_labels.items()]
knn.fit(sample, target)




