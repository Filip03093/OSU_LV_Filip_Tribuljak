import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv('data_C02_emission.csv')

numeric_columns = [
    'Engine Size (L)',
    'Cylinders',
    'Fuel Consumption City (L/100km)',
    'Fuel Consumption Hwy (L/100km)',
    'Fuel Consumption Comb (L/100km)',
    'Fuel Consumption Comb (mpg)'
]
output_column = 'CO2 Emissions (g/km)'

df_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

ohe = OneHotEncoder(sparse_output=False)
fuel_train = ohe.fit_transform(df_train[['Fuel Type']])
fuel_test = ohe.transform(df_test[['Fuel Type']])

X_train = np.hstack([df_train[numeric_columns].values, fuel_train])
X_test = np.hstack([df_test[numeric_columns].values, fuel_test])

y_train = df_train[output_column].values
y_test = df_test[output_column].values

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

MSE = mean_squared_error(y_test, y_pred)
RMSE = MSE ** 0.5
MAE = mean_absolute_error(y_test, y_pred)
R2  = r2_score(y_test, y_pred)
MAPE = np.mean(np.abs(y_test - y_pred) / np.maximum(1e-10, np.abs(y_test)))

print(f"MSE = {MSE:.4f}")
print(f"RMSE = {RMSE:.4f}")
print(f"MAE = {MAE:.4f}")
print(f"R2  = {R2:.4f}")
print(f"MAPE = {MAPE:.4f}")

errors = np.abs(y_test - y_pred)
max_idx = np.argmax(errors)

print(f"\nMaksimalna greska: {errors[max_idx]:.2f} g/km")
print(f"Vozilo: {df_test.iloc[max_idx]['Make']} {df_test.iloc[max_idx]['Model']}")
print(f"Stvarna emisija:  {y_test[max_idx]} g/km")
print(f"Procjena modela: {y_pred[max_idx]:.2f} g/km")

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.xlabel('Stvarne vrijednosti CO2 (g/km)')
plt.ylabel('Procjena modela CO2 (g/km)')
plt.title('Stvarne vrijednosti vs. procjena modela')
plt.grid(True)
plt.show()
