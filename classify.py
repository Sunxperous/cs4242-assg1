from sklearn import neighbors
from sklearn import svm

from index import Index


index = Index('testing')

sample = [v for k, v in index.tweet_features.items()]
target = [v for k, v in index.development_labels.items()]

knn = neighbors.KNeighborsClassifier(weights='uniform')
knn.fit(sample, target)

svc = svm.SVC(kernel='linear')
# linear kernel is used when feature size is big (~10,000) and sample size is moderate (5000)
# C is default 1.0, decrease if overfitting, increase if underfitting
# if uneven data result, try adding (class_weight='balanced')
# scaling and normalization is highly recommended
svc.fit(sample, target) 




