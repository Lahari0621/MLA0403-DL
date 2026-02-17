import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Split
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=1)

# Model
pred = RandomForestClassifier(n_estimators=100, random_state=1)\
       .fit(Xtr, ytr).predict(Xte)

# Accuracy
print("Accuracy:", accuracy_score(yte, pred))

# Confusion Matrix
sns.heatmap(confusion_matrix(yte, pred),
            annot=True, fmt='d', cmap='Greens',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - Random Forest")
plt.show()
