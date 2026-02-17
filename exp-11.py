import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Split
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=1)

# Train model
model = DecisionTreeClassifier()
pred = model.fit(Xtr, ytr).predict(Xte)

# Accuracy
print("Accuracy:", accuracy_score(yte, pred))

# Confusion Matrix
sns.heatmap(confusion_matrix(yte, pred),
            annot=True, fmt='d', cmap='PRGn',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Decision Tree Visualization
plt.figure(figsize=(12, 6))
plot_tree(model,
          feature_names=iris.feature_names,
          class_names=iris.target_names,
          filled=True)
plt.title("Decision Tree")
plt.show()
