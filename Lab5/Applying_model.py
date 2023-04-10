import sklearn
# https://scikit-learn.org/stable/datasets/index.html#breast-cancer-wisconsin-diagnostic-database
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
label_names = data['target_names']
labels = data['target']
feature_names = data['feature_names']
features = data['data']

print(label_names)
print(labels[0], ', ', labels[1])
print(feature_names[0], ', ', feature_names[1])
print(features[0], ', ', features[1])

from sklearn.model_selection import train_test_split

train, test, train_labels, test_labels = \
train_test_split(features,labels,test_size = 0.40, random_state = 42)

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
model = gnb.fit(train, train_labels)
preds = gnb.predict(test)
print(preds)

from sklearn.metrics import accuracy_score

print(accuracy_score(test_labels,preds))
