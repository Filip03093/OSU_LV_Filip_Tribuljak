import pandas as pd
import numpy as np

data = pd.read_csv('data_C02_emission.csv')

print("a)\n")

print(f"Broj mjerenja: {len(data)}")
print(f"\nTipovi varijabli:\n{data.dtypes}")

print(f"\nIzostale vrijednosti po stupcu:\n{data.isnull().sum()}")
print(f"\nBroj dupliciranih redaka: {data.duplicated().sum()}")

data = data.dropna(axis=0)
data = data.drop_duplicates()
data = data.reset_index(drop=True)

categorical_cols = ['Make', 'Model', 'Vehicle Class', 'Transmission', 'Fuel Type']
for col in categorical_cols:
    data[col] = data[col].astype('category')

print(f"\nTipovi nakon konverzije:\n{data.dtypes}")


print("\nb)\n")

cols = ['Make', 'Model', 'Fuel Consumption City (L/100km)']

najmanji = data.sort_values(by='Fuel Consumption City (L/100km)').head(3)
najveci = data.sort_values(by='Fuel Consumption City (L/100km)').tail(3)

print("3 vozila s najmanjom gradskom potrošnjom:")
print(najmanji[cols].to_string(index=False))
print("\n3 vozila s najvećom gradskom potrošnjom:")
print(najveci[cols].to_string(index=False))


print("\nc)\n")

motor_filter = data[
    (data['Engine Size (L)'] >= 2.5) & (data['Engine Size (L)'] <= 3.5)
]
print(f"Broj vozila: {len(motor_filter)}")
print(f"Prosječna CO2 emisija tih vozila: {motor_filter['CO2 Emissions (g/km)'].mean():.2f} g/km")


print("\nd)\n")

audi = data[data['Make'] == 'Audi']
print(f"Ukupan broj Audi mjerenja: {len(audi)}")

audi_4cil = audi[audi['Cylinders'] == 4]
print(f"Prosječna CO2 emisija Audi s 4 cilindra: {audi_4cil['CO2 Emissions (g/km)'].mean():.2f} g/km")


print("\ne)\n")

grouped_cil = data.groupby('Cylinders')

print("Broj vozila po broju cilindara:")
print(grouped_cil.size().to_string())

print("\nProsječna CO2 emisija po broju cilindara:")
print(grouped_cil['CO2 Emissions (g/km)'].mean().round(2).to_string())


print("\nf)\n")

dizel = data[data['Fuel Type'] == 'D']
benzin = data[data['Fuel Type'] == 'X']

print(f"Dizel – prosječna gradska potrošnja: {dizel['Fuel Consumption City (L/100km)'].mean():.2f} L/100km")
print(f"Dizel – medijan gradske potrošnje: {dizel['Fuel Consumption City (L/100km)'].median():.2f} L/100km")

print(f"\nBenzin – prosječna gradska potrošnja: {benzin['Fuel Consumption City (L/100km)'].mean():.2f} L/100km")
print(f"Benzin – medijan gradske potrošnje: {benzin['Fuel Consumption City (L/100km)'].median():.2f} L/100km")



print("\ng)\n")

filter_g = data[(data['Cylinders'] == 4) & (data['Fuel Type'] == 'D')]
worst = filter_g.sort_values(by='Fuel Consumption City (L/100km)').tail(1)

print(worst[['Make', 'Model', 'Fuel Consumption City (L/100km)']].to_string(index=False))


print("\nh)\n")

rucni = data[data['Transmission'].str.startswith('M')]
print(f"Broj vozila s ručnim mjenjačem: {len(rucni)}")


print("\ni)\n")

korelacija = data.corr(numeric_only=True)
print(korelacija.round(3))
