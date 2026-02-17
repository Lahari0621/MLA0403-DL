import numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Data + model
X, y = make_classification(n_samples=100, n_features=2, n_informative=2,
                           n_redundant=0, n_clusters_per_class=1, random_state=42)
model = LogisticRegression().fit(X, y)

# Accuracy
y_pred = model.predict(X)
print(f"Accuracy: {accuracy_score(y, y_pred):.2f}")

# Plot
plt.scatter(X[:,0], X[:,1], c=y, cmap='berlin', edgecolor='m', s=50)

coef, intercept = model.coef_[0], model.intercept_
x_vals = np.linspace(X[:,0].min(), X[:,0].max(), 100)
plt.plot(x_vals, -(coef[0]*x_vals + intercept)/coef[1], 'k--')

plt.xlabel('Feature 1'); plt.ylabel('Feature 2')
plt.title('Linear Separability Demonstration')
plt.show()
