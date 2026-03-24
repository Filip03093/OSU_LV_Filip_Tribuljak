import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

df = pd.read_csv('data_C02_emission.csv')

input_columns = [
    'Engine Size (L)',
    'Cylinders',
    'Fuel Consumption City (L/100km)',
    'Fuel Consumption Hwy (L/100km)',
    'Fuel Consumption Comb (L/100km)',
    'Fuel Consumption Comb (mpg)'
]
output_column = 'CO2 Emissions (g/km)'

X = df[input_columns].values
y = df[output_column].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)


feature_index = input_columns.index('Engine Size (L)')

plt.figure(figsize=(8, 6))
plt.scatter(X_train[:, feature_index], y_train, color='blue', label='Trening skup')
plt.scatter(X_test[:, feature_index], y_test, color='red', label='Testni skup')

plt.xlabel('Engine Size (L)')
plt.ylabel('CO2 Emissions (g/km)')
plt.title('Ovisnost emisije CO2 o velicini motora')
plt.legend()
plt.show()


sc = StandardScaler()

X_train_s = sc.fit_transform(X_train)
X_test_s = sc.transform(X_test)

feature_index = input_columns.index('Engine Size (L)')

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(X_train[:, feature_index], bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Engine Size (L)')
plt.ylabel('Frekvencija')
plt.title('Prije standardizacije')

plt.subplot(1, 2, 2)
plt.hist(X_train_s[:, feature_index], bins=20, color='orange', edgecolor='black')
plt.xlabel('Standardizirana vrijednost')
plt.ylabel('Frekvencija')
plt.title('Nakon standardizacije')

plt.tight_layout()
plt.show()


model = LinearRegression()
model.fit(X_train_s, y_train)

print("theta0 =", model.intercept_)

for i in range(len(model.coef_)):
    print(f"theta{i+1} = {model.coef_[i]}")


y_test_pred = model.predict(X_test_s)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test_pred, color='blue')
plt.xlabel('Stvarne vrijednosti CO2 (g/km)')
plt.ylabel('Procjena modela CO2 (g/km)')
plt.title('Stvarne vrijednosti vs. procjena modela')
plt.show()


MSE = mean_squared_error(y_test, y_test_pred)
RMSE = MSE ** 0.5
MAE = mean_absolute_error(y_test, y_test_pred)
R2 = r2_score(y_test, y_test_pred)
MAPE = np.mean(np.abs(y_test - y_test_pred) / np.maximum(1e-10, np.abs(y_test)))

print(f"MSE = {MSE:.4f}")
print(f"RMSE = {RMSE:.4f}")
print(f"MAE = {MAE:.4f}")
print(f"R2 = {R2:.4f}")
print(f"MAPE = {MAPE:.4f}")