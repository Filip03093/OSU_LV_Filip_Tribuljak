import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('data.csv', delimiter=',', skip_header=1)

n_people = data.shape[0]
print(f"Number of people: {n_people}")

ind_m = data[:, 0] == 1
ind_f = data[:, 0] == 0

plt.scatter(data[ind_m, 1], data[ind_m, 2], color='blue', s=5, alpha=0.4, label='Male')
plt.scatter(data[ind_f, 1], data[ind_f, 2], color='red', s=5, alpha=0.4, label='Female')
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.legend()
plt.show()

ind_50 = np.arange(0, data.shape[0], 50)
data_50 = data[ind_50, :]
ind_m50 = data_50[:, 0] == 1
ind_f50 = data_50[:, 0] == 0
plt.scatter(data_50[ind_m50, 1], data_50[ind_m50, 2], s=8, alpha=0.7, color='blue', label='Male')
plt.scatter(data_50[ind_f50, 1], data_50[ind_f50, 2], s=8, alpha=0.7, color='red',  label='Female')
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.legend()
plt.show()

print(data[:, 1].min(), data[:, 1].max(), data[:, 1].mean())

males = data[ind_m, :]
females = data[ind_f, :]

print(males[:, 1].min(), males[:, 1].max(), males[:, 1].mean())
print(females[:, 1].min(), females[:, 1].max(), females[:, 1].mean())
