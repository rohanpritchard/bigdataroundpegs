from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

# Load dataset
digits = load_digits()
print(digits.data.shape)

# Visualise point
plt.gray()
plt.matshow(digits.images[98])  # Enter datapoint index here
plt.show()
