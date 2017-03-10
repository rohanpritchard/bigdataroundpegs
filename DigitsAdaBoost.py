from __future__ import division
from sklearn import datasets
from sklearn.ensemble import AdaBoostClassifier


digits = datasets.load_digits()
Xtrain = digits.data[::2]
Ytrain = digits.target [::2]
Xtest = digits.data[1::2]
Ytest = digits.target[1::2]

ada = AdaBoostClassifier()
ada.fit(Xtrain, Ytrain)
results = ada.predict(Xtest)
answers = Ytest

c = zip(results, answers)
correct = 0
total = 0

for (a,b) in c:
    if a==b:
        correct+=1
    total+=1

print "Correctly identified %s out of %s data points. Success rate of: %s." % (correct, total, round(correct/total, 3))