import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from io import BytesIO

iris = load_iris()
X, y = iris.data, iris.target

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=0)

clf = RidgeClassifier(tol=1e-2, solver="sag")
clf.fit(Xtrain, ytrain)

ypred = clf.predict(Xtest)

print(f"Accuracy: {np.round(metrics.accuracy_score(ytest, ypred), 4)}")
print(f"Precision: {np.round(metrics.precision_score(ytest, ypred, average='weighted'), 4)}")
print(f"Recall: {np.round(metrics.recall_score(ytest, ypred, average='weighted'), 4)}")
print(f"F1 Score: {np.round(metrics.f1_score(ytest, ypred, average='weighted'), 4)}")
print(f"Cohen Kappa Score: {np.round(metrics.cohen_kappa_score(ytest, ypred), 4)}")
print(f"Matthews Corrcoef: {np.round(metrics.matthews_corrcoef(ytest, ypred), 4)}")

print('\nClassification Report:\n', metrics.classification_report(ytest, ypred))

sns.set()
mat = metrics.confusion_matrix(ytest, ypred)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label') 

plt.savefig("Confusion.jpg")
plt.show()