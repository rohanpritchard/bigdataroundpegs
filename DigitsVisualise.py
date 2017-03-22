from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

# Load dataset
digits = load_digits()
print(digits.data.shape)

# Enter data point to view:
x = 98

# Print point values
print digits.images[x]

# Visualise point
plt.gray()
plt.matshow(digits.images[x])
plt.show()
