import pandas as pd
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import numpy as np

# КРОК 1
# Виправлено посилання з .esv на .csv
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(url, names=names)

print("КРОК 1: Дослідження даних")
print("Форма датасету:", dataset.shape)
print("\nСтатистичне зведення:\n", dataset.describe())
print("\nРозподіл за класами:\n", dataset.groupby('class').size())

# КРОК 2
# УВАГА: Програма буде зупинятися на кожному графіку. 
# Щоб код пішов далі, вам потрібно закрити вікно з графіком!
print("\nКРОК 2: Будуємо графіки (закрийте вікно графіка, щоб продовжити)")

dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.suptitle("Діаграма розмаху (Box and Whisker)")
plt.show()

dataset.hist()
plt.suptitle("Гістограма розподілу")
plt.show()

scatter_matrix(dataset)
plt.suptitle("Матриця діаграм розсіювання")
plt.show()

# КРОК 3
array = dataset.values
X = array[:, 0:4]
y = array[:, 4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1, shuffle=True)

# КРОК 4
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

print("\nКРОК 4: Порівняння алгоритмів (Крос-валідація)")
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# Графік порівняння алгоритмів
plt.boxplot(results, labels=names)
plt.title('Algorithm Comparison')
plt.show()

# КРОК 6-7
print("\nКРОК 7: Оцінка якості моделі SVM на контрольній вибірці")
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

print("Accuracy (Точність):", accuracy_score(Y_validation, predictions))
print("\nМатриця помилок:\n", confusion_matrix(Y_validation, predictions))
print("\nЗвіт про класифікацію:\n", classification_report(Y_validation, predictions))

# КРОК 8
print("\nКРОК 8: Прогноз для знайденого ірису")
X_new = np.array([[5.0, 2.9, 1.0, 0.2]])
print("Форма масиву X_new:", X_new.shape)

prediction = model.predict(X_new)
print(f"Спрогнозований сорт ірису: {prediction[0]}")