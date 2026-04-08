import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score

# --- Підготовка даних ---
input_file = 'income_data.txt'
X, y = [], []
count_class1, count_class2 = 0, 0
max_datapoints = 25000

with open(input_file, 'r') as f:
    for line in f.readlines():
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints: break
        if '?' in line: continue
        data = line[:-1].split(', ')
        if data[-1] == '<=50K' and count_class1 < max_datapoints:
            X.append(data)
            count_class1 += 1
        if data[-1] == '>50K' and count_class2 < max_datapoints:
            X.append(data)
            count_class2 += 1

X = np.array(X)
label_encoder = []
X_encoded = np.empty(X.shape)

for i, item in enumerate(X[0]):
    if item.isdigit():
        X_encoded[:, i] = X[:, i]
    else:
        le = preprocessing.LabelEncoder()
        label_encoder.append(le)
        X_encoded[:, i] = le.fit_transform(X[:, i])

X_data = X_encoded[:, :-1].astype(int)
y_data = X_encoded[:, -1].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=5)

# --- Навчання та оцінка (Поліноміальне ядро) ---
print("--- Поліноміальне ядро (Poly) ---")
classifier = OneVsOneClassifier(SVC(kernel='poly', degree=2, random_state=0))
classifier.fit(X_train, y_train)

y_test_pred = classifier.predict(X_test)
f1 = cross_val_score(classifier, X_data, y_data, scoring='f1_weighted', cv=3)

print("F1 score (CV):", round(100 * f1.mean(), 2), "%")
print("Accuracy:", round(accuracy_score(y_test, y_test_pred) * 100, 2), "%")
print("Precision:", round(precision_score(y_test, y_test_pred, average='weighted', zero_division=0) * 100, 2), "%")
print("Recall:", round(recall_score(y_test, y_test_pred, average='weighted') * 100, 2), "%")