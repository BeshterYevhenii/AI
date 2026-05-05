import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV # Оновлено з cross_validation та grid_search
from sklearn.ensemble import ExtraTreesClassifier
from utilities import visualize_classifier

if __name__ == '__main__':
    input_file = 'data_random_forests.txt'
    data = np.loadtxt(input_file, delimiter=',')
    X, y = data[:, :-1], data[:, -1]

    # Розбиття даних на навчальний та тестовий набори
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)

    # Визначення сітки значень параметрів
    parameter_grid = [
        {'n_estimators': [100], 'max_depth': [2, 4, 7, 12, 16]},
        {'max_depth': [4], 'n_estimators': [25, 50, 100, 250]}
    ]

    metrics = ['precision_weighted', 'recall_weighted']

    for metric in metrics:
        print("\n##### Searching optimal parameters for", metric)
        # Оновлено використання GridSearchCV
        classifier = GridSearchCV(ExtraTreesClassifier(random_state=0), parameter_grid, cv=5, scoring=metric)
        classifier.fit(X_train, y_train)

        print("\nGrid scores for the parameter grid:")
        means = classifier.cv_results_['mean_test_score']
        params = classifier.cv_results_['params']
        for mean, param in zip(means, params):
            print(param, '-->', round(mean, 3))

        print("\nBest parameters:", classifier.best_params_)

        y_pred = classifier.predict(X_test)
        print("\nPerformance report:\n")
        print(classification_report(y_test, y_pred))