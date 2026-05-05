import datetime
import json
import numpy as np
from sklearn import cluster
from sklearn.covariance import GraphicalLassoCV
import yfinance as yf

# Створення словника прив'язок символів компаній до їх повних назв
company_symbols_map = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corp.",
    "IBM": "Microsoft Corp.", 
    "XOM": "Intl Business Machines",
    "CVX": "Exxon Mobil Corp.",
    "INTC": "Intel Corp.",
    "JNJ": "Johnson & Johnson",
    "JPM": "JPMorgan Chase & Co.",
    "WMT": "Wal-Mart Stores",
    "PG": "Procter & Gamble Co."
}

symbols = np.array(list(company_symbols_map.keys()))
names = np.array(list(company_symbols_map.values()))

# Завантаження архівних даних котирувань
start_date = "2003-07-03"
end_date = "2007-05-04"

print("Завантаження даних...")
opening_quotes = []
closing_quotes = []

for symbol in symbols:
    data = yf.download(symbol, start=start_date, end=end_date, progress=False)
    opening_quotes.append(data['Open'].values.flatten())
    closing_quotes.append(data['Close'].values.flatten())

# Вилучення котирувань та приведення до однієї довжини 
min_len = min([len(q) for q in opening_quotes])
opening_quotes = np.array([q[:min_len] for q in opening_quotes]).astype(float)
closing_quotes = np.array([q[:min_len] for q in closing_quotes]).astype(float)

# Обчислення різниці між двома видами котирувань
quotes_diff = closing_quotes - opening_quotes

# Нормалізація даних
X = quotes_diff.copy().T
X /= X.std(axis=0)

# Створення моделі графа 
edge_model = GraphicalLassoCV()

# Навчання моделі
print("Навчання моделі...")
with np.errstate(invalid='ignore'):
    edge_model.fit(X)

# Створення моделі кластеризації на основі поширення подібності
_, labels = cluster.affinity_propagation(edge_model.covariance_, random_state=0)
num_labels = labels.max()

# Виведення результатів
print("\nРЕЗУЛЬТАТИ КЛАСТЕРИЗАЦІЇ ФОНДОВОГО РИНКУ:")
for i in range(num_labels + 1):
    print(f"Cluster {i+1} ==> {', '.join(names[labels == i])}")