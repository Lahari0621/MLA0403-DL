import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)

pred = GaussianNB().fit(Xtr, ytr).predict(Xte)

print("Accuracy:", accuracy_score(yte, pred))

sns.heatmap(confusion_matrix(yte, pred),
            annot=True, fmt='d', cmap='viridis',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - Naive Bayes')
plt.show()
