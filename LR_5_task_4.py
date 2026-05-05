import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.datasets import fetch_california_housing # Використано замість проблемного load_boston
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    # Завантаження даних із цінами на нерухомість (Каліфорнія)
    housing_data = fetch_california_housing()

    # Перемішування даних для об'єктивності аналізу
    X, y = shuffle(housing_data.data, housing_data.target, random_state=7)

    # Розбиття даних на навчальний та тестовий набори (20% на тест)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

    # Визначення і навчання регресора AdaBoost
    # Використовується DecisionTreeRegressor як базова індивідуальна модель
    regressor = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=400, random_state=7)
    regressor.fit(X_train, y_train)

    # Обчислення показників ефективності (MSE та EVS)
    y_pred = regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    evs = explained_variance_score(y_test, y_pred)
    
    print("\nADABOOST REGRESSOR")
    print("Mean squared error =", round(mse, 2))
    print("Explained variance score =", round(evs, 2))

    # Вилучення відносної важливості кожної ознаки
    feature_importances = regressor.feature_importances_
    feature_names = housing_data.feature_names

    # Нормалізація значень важливості ознак (щоб максимум був 100)
    feature_importances = 100.0 * (feature_importances / max(feature_importances))
    
    # Сортування та перестановка значень для побудови графіка від найбільшого до найменшого
    index_sorted = np.flipud(np.argsort(feature_importances))
    
    # Розміщення міток уздовж осі Х
    pos = np.arange(index_sorted.shape[0]) + 0.5

    # Побудова стовпчастої діаграми
    plt.figure()
    plt.bar(pos, feature_importances[index_sorted], align='center')
    plt.xticks(pos, np.array(feature_names)[index_sorted], rotation=90)
    plt.ylabel('Relative Importance')
    plt.title('Оцінка важливості ознак з використанням регресора AdaBoost')
    plt.tight_layout() # Автоматично підлаштовує відступи, щоб мітки не обрізались
    plt.show()