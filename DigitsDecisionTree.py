from __future__ import division
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier

# Load dataset
digits = datasets.load_digits()

# Split dataset in half; one for training, one for testing.
Xtrain = digits.data[::2]
Ytrain = digits.target [::2]
Xtest = digits.data[1::2]
Ytest = digits.target[1::2]

# Train model
dt = DecisionTreeClassifier()
dt.fit(Xtrain,Ytrain)

# Predict test data
results = dt.predict(Xtest)
answers = Ytest

# Compare results
guessAnswerPair = zip(results, answers)
correct = 0
total = 0
for (guess, answer) in guessAnswerPair:
    if guess==answer:
        correct+=1
    total+=1

print "Correctly identified %s out of %s data points. Success rate of: %s." % (correct, total, round(correct/total, 3))