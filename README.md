# The Round Pegs and Square Holes of Big Data - Code Scripts
The following scripts have been written in python, and correspond to various examples written on our website on big data. In these examples I have made use of MatPlotLib, and SciKit. I have also made use of the sample datasets available with SciKit.

## Basic Regression Examples
Referred to at: [LINK HERE]. Shows the advantage in some instances of implementing random forests. When working with noisy data.

### BasicRegression.py
Demonstrates the use of SciKit's DecisionTreeRegressor to predict sin values. Also makes use of graphviz to export the decision tree and pyplot to show the predictions.

### RandomForestRegression.py
Demonstrates the use of SciKit's RandomForestRegressor to predict sin values more effectively than with a standard decision tree. Uses pyplot to show the predictions.

## Case Study - Digits Dataset
Referred to at: [LINK HERE]. Demonstrates another, more interesting example of improving the performance of a standard decision tree classifier with higher dimensioned noisy data.

### DigitsDecisionTree.py
Demonstrates a standard DecisionTreeClassifier from SciKit in order to build a handwriting recognition model.

### DigitsRandomForest.py
Demonstrates a RandomForestClassifier from SciKit which improves on the accuracy of the previous standard decision tree model.

### DigitsVisualise.py
Allows the user to view datapoints of the 64 dimensional data as a picture (via Pyplot), to associate to a handwritten digit. Can be useful in order to analyse the quality of the data manually.
