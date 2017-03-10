import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Create a random dataset
rng = np.random.RandomState(3)
X = np.sort(5 * rng.rand(300, 1), axis=0)
y = np.sin(X).ravel()
trainingX = X[::2]
testingX = X[1::2]
trainingY = y[::2]
testingY = y[1::2]
trainingY[::5] += 3 * (0.5 - rng.rand(30))

# Fit regression model
rf = RandomForestRegressor(n_estimators=20, max_depth=4)
rf.fit(trainingX, trainingY)

# Predict
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
y_2 = rf.predict(X_test)

# Test with test data
testResults = rf.predict(testingX)
acc = 0
for i in range(0, len(testResults)):
    acc += abs(testResults[i] - testingY[i])
averageError = acc / len(testResults)
print averageError


# Plot the results
plt.figure()
plt.scatter(X, y, c="black", label="data")
plt.plot(X_test, y_2, color="red", label="max_depth=4", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Random Forest Regression")
plt.legend()
plt.show()
