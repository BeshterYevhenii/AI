import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Вхідний файл, який містить дані
input_file = 'income_data.txt'

# Читання даних
X = []
y = []
count_class1 = 0
count_class2 = 0
max_datapoints = 25000

# Відкриємо файл і прочитаємо рядки
with open(input_file, 'r') as f:
    for line in f.readlines():
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break
        if '?' in line: # Пропускаємо рядки з відсутніми даними
            continue
            
        # Кожен рядок даних відокремлюється від наступного за допомогою коми
        data = line[:-1].split(', ')
        
        if data[-1] == '<=50K' and count_class1 < max_datapoints:
            X.append(data)
            count_class1 += 1
        if data[-1] == '>50K' and count_class2 < max_datapoints:
            X.append(data)
            count_class2 += 1

# Перетворення на масив numpy
X = np.array(X)

# Перетворення рядкових даних на числові
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

# Створення SVM-класифікатора з лінійним ядром
# Додано dual=False та max_iter, щоб уникнути попереджень про збіжність (ConvergenceWarning)
classifier = OneVsOneClassifier(LinearSVC(random_state=0, dual=False, max_iter=10000))

# Навчання класифікатора
classifier.fit(X_data, y_data)

# Виконайте перехресну перевірку, розбивши дані на навчальний та тестовий набори (80/20)
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=5)

classifier.fit(X_train, y_train)
y_test_pred = classifier.predict(X_test)

# Обчислення F-міри для SVM-класифікатора через крос-валідацію
f1 = cross_val_score(classifier, X_data, y_data, scoring='f1_weighted', cv=3)
print("F1 score (cross-validation): " + str(round(100 * f1.mean(), 2)) + "%")

# Обчислення інших показників якості класифікації (завдання з методички)
print("Accuracy (Акуратність):", round(accuracy_score(y_test, y_test_pred) * 100, 2), "%")
print("Precision (Точність):", round(precision_score(y_test, y_test_pred, average='weighted') * 100, 2), "%")
print("Recall (Повнота):", round(recall_score(y_test, y_test_pred, average='weighted') * 100, 2), "%")
print("-" * 40)

# Передбачення результату для тестової точки даних
input_data = ['37', 'Private', '215646', 'HS-grad', '9', 'Never-married', 
              'Handlers-cleaners', 'Not-in-family', 'White', 'Male', 
              '0', '0', '40', 'United-States']

# Кодування тестової точки даних
input_data_encoded = [-1] * len(input_data)
count = 0

for i, item in enumerate(input_data):
    if item.isdigit():
        input_data_encoded[i] = int(input_data[i])
    else:
        # Для transform потрібно передавати масив
        input_data_encoded[i] = int(label_encoder[count].transform([input_data[i]])[0])
        count += 1

# Зміна форми масиву для одного передбачення (моделі вимагають 2D масив)
input_data_encoded = np.array(input_data_encoded).reshape(1, -1)

# Використання класифікатора для кодованої точки даних та виведення результату
predicted_class = classifier.predict(input_data_encoded)
print("Прогноз для тестової точки (рівень доходу):", label_encoder[-1].inverse_transform(predicted_class)[0])
