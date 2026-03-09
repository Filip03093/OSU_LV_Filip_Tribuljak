import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data_C02_emission.csv')

data = data.dropna(axis=0)
data = data.drop_duplicates()
data = data.reset_index(drop=True)
categorical_cols = ['Make', 'Model', 'Vehicle Class', 'Transmission', 'Fuel Type']
for col in categorical_cols:
    data[col] = data[col].astype('category')

fuel_colors = {'X': 'blue', 'Z': 'red', 'D': 'green', 'E': 'orange', 'N': 'purple'}


plt.figure()
data['CO2 Emissions (g/km)'].plot(kind='hist', bins=20, edgecolor='black')
plt.title('Distribucija CO2 emisije')
plt.xlabel('CO2 emisija (g/km)')
plt.ylabel('Broj vozila')
plt.tight_layout()


plt.figure()
for fuel, color in fuel_colors.items():
    subset = data[data['Fuel Type'] == fuel]
    plt.scatter(subset['Fuel Consumption City (L/100km)'], subset['CO2 Emissions (g/km)'], c=color, label=fuel, alpha=0.5, s=20)
plt.title('Gradska potrosnja vs CO2 emisija')
plt.xlabel('Gradska potrosnja (L/100km)')
plt.ylabel('CO2 emisija (g/km)')
plt.legend(title='Tip goriva')
plt.tight_layout()


data.boxplot(column='Fuel Consumption Hwy (L/100km)', by='Fuel Type')
plt.title('Izvangradska potrosnja po tipu goriva')
plt.suptitle('')
plt.xlabel('Tip goriva')
plt.ylabel('Izvangradska potrosnja (L/100km)')
plt.tight_layout()


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

data.groupby('Fuel Type').size().plot(kind='bar', edgecolor='black', ax=ax1)
ax1.set_title('Broj vozila po tipu goriva')
ax1.set_xlabel('Tip goriva')
ax1.set_ylabel('Broj vozila')
ax1.tick_params(axis='x', rotation=0)

data.groupby('Cylinders')['CO2 Emissions (g/km)'].mean().round(2).plot(kind='bar', edgecolor='black', ax=ax2)
ax2.set_title('Prosjecna CO2 emisija po broju cilindara')
ax2.set_xlabel('Broj cilindara')
ax2.set_ylabel('Prosjecna CO2 emisija (g/km)')
ax2.tick_params(axis='x', rotation=0)

plt.tight_layout()

plt.show()
