import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

m = 100
np.random.seed(42) 
X = 6 * np.random.rand(m, 1) - 5
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)

def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=10)
    train_errors, val_errors = [], []
    for m_val in range(1, len(X_train) + 1):
        model.fit(X_train[:m_val], y_train[:m_val])
        y_train_predict = model.predict(X_train[:m_val])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m_val], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))
    
    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="Навчальний набір")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="Перевірочний набір")
    plt.xlabel("Розмір навчального набору")
    plt.ylabel("RMSE")
    plt.legend()

plt.figure(figsize=(10, 4))
lin_reg = LinearRegression()
plot_learning_curves(lin_reg, X, y)
plt.title("Криві навчання для лінійної моделі")
plt.ylim(0, 3)
plt.show()

plt.figure(figsize=(10, 4))
polynomial_regression_10 = Pipeline([
    ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
    ("lin_reg", LinearRegression())
])
plot_learning_curves(polynomial_regression_10, X, y)
plt.title("Криві навчання для поліноміальної моделі (ступінь 10)")
plt.ylim(0, 3)
plt.show()

plt.figure(figsize=(10, 4))
polynomial_regression_2 = Pipeline([
    ("poly_features", PolynomialFeatures(degree=2, include_bias=False)),
    ("lin_reg", LinearRegression())
])
plot_learning_curves(polynomial_regression_2, X, y)
plt.title("Криві навчання для поліноміальної моделі (ступінь 2)")
plt.ylim(0, 3)
plt.show()