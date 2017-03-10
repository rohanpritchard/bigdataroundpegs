print(__doc__)

# Import the necessary modules and libraries
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

# Create a random dataset
rng = np.random.RandomState(1)
X = np.sort(5 * rng.rand(300, 1), axis=0)
y = np.sin(X).ravel()
y[::5] += 3 * (0.5 - rng.rand(60))
y[::2] += 0.5*(0.5 - rng.rand(150))

# Fit regression model
randomForest = RandomForestRegressor(max_depth=3, n_estimators=100)
basicTree = DecisionTreeRegressor(max_depth=3)
randomForest.fit(X, y)
basicTree.fit(X, y)

# Predict
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
randForestY = randomForest.predict(X_test)
basicTreeY = basicTree.predict(X_test)

print basicTree.predict(3.14)
print randomForest.predict(3.14)

# Plot the results
plt.figure()
plt.scatter(X, y, c="darkorange", label="data")
plt.plot(X_test, randForestY, color="cornflowerblue", label="Random Forest", linewidth=1)
plt.plot(X_test, basicTreeY, color="yellowgreen", label="Basic Tree", linewidth=1)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Random Forest Regression")
plt.legend()
plt.show()


