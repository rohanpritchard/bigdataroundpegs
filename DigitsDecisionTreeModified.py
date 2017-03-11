from __future__ import division
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

def visualise(x):
    plt.gray()
    plt.matshow(digits.images[x])
    plt.show()

digits = datasets.load_digits()
Xtrain = digits.data[::2]
Ytrain = digits.target [::2]
Xtest = digits.data[1::2]
Ytest = digits.target[1::2]

dt = DecisionTreeClassifier()
dt.fit(Xtrain,Ytrain)
results = dt.predict(Xtest)
answers = Ytest


guessAnswerPair = zip(results, answers)
correct = 0
total = 0
incorrect = []

for (guess, answer) in guessAnswerPair:
    if guess==answer:
        correct+=1
    else:
        toAdd = (total, (guess, answer))
        incorrect.append(toAdd)
    total+=1

print incorrect
visualise(incorrect[3][0])
print "Correctly identified %s out of %s data points. Success rate of: %s." % (correct, total, round(correct/total, 3))