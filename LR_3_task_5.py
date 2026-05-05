import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

m = 100
X = 6 * np.random.rand(m, 1) - 5
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)

lin_reg = LinearRegression()
lin_reg.fit(X, y)

poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)

print("Отримана поліноміальна модель")
print(f"y = {poly_reg.coef_[0][1]:.2f} * X^2 + {poly_reg.coef_[0][0]:.2f} * X + {poly_reg.intercept_[0]:.2f}")

X_new = np.linspace(-5, 1, 100).reshape(100, 1)
y_new_lin = lin_reg.predict(X_new)
X_new_poly = poly_features.transform(X_new)
y_new_poly = poly_reg.predict(X_new_poly)

plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='gray', s=15, label='Випадкові дані')
plt.plot(X_new, y_new_lin, color='red', label='Лінійна регресія')
plt.plot(X_new, y_new_poly, color='green', linewidth=2, label='Поліноміальна регресія (ступінь 2)')
plt.xlabel("$X_1$")
plt.ylabel("y")
plt.legend()
plt.title("Порівняння лінійної та поліноміальної регресії")
plt.show()