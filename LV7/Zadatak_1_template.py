import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.cluster import KMeans, AgglomerativeClustering


def generate_data(n_samples, flagc):
    # 3 grupe
    if flagc == 1:
        random_state = 365
        X,y = make_blobs(n_samples=n_samples, random_state=random_state)
    
    # 3 grupe
    elif flagc == 2:
        random_state = 148
        X,y = make_blobs(n_samples=n_samples, random_state=random_state)
        transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
        X = np.dot(X, transformation)

    # 4 grupe 
    elif flagc == 3:
        random_state = 148
        X, y = make_blobs(n_samples=n_samples,
                        centers = 4,
                        cluster_std=np.array([1.0, 2.5, 0.5, 3.0]),
                        random_state=random_state)
    # 2 grupe
    elif flagc == 4:
        X, y = make_circles(n_samples=n_samples, factor=.5, noise=.05)
    
    # 2 grupe  
    elif flagc == 5:
        X, y = make_moons(n_samples=n_samples, noise=.05)
    
    else:
        X = []
        
    return X



fig, axes = plt.subplots(1, 5, figsize=(20, 4))

for i, flagc in enumerate(range(1, 6)):
    X = generate_data(500, flagc)

    axes[i].scatter(X[:, 0], X[:, 1], s=10)
    axes[i].set_xlabel('$x_1$')
    axes[i].set_ylabel('$x_2$')
    axes[i].set_title(f'flagc={flagc}')

plt.suptitle('Podatkovni primjeri za sve flagc vrijednosti')
plt.tight_layout()
plt.show()



fig, axes = plt.subplots(1, 5, figsize=(20, 4))
flagc = 1
X = generate_data(500, flagc)
for i, K in enumerate(range(1, 6)):

    km = KMeans(n_clusters=K, random_state=0)
    km.fit(X)

    labels = km.labels_
    centers = km.cluster_centers_

    axes[i].scatter(X[:, 0], X[:, 1], c=labels, cmap='tab10', s=10)
    axes[i].scatter(centers[:, 0], centers[:, 1], c='black', marker='X', s=100, label='Centri')
    axes[i].set_xlabel('$x_1$')
    axes[i].set_ylabel('$x_2$')
    axes[i].set_title(f'K = {K}')
    axes[i].legend()

plt.suptitle(f'K-Means grupiranje za K = 1 do 5 (flagc={flagc})', fontsize=13)
plt.tight_layout()
plt.show()



inertias = []
K_range = range(1, 11)

for k in K_range:
    km_test = KMeans(n_clusters=k, random_state=0)
    km_test.fit(X)
    inertias.append(km_test.inertia_)

plt.figure()
plt.plot(list(K_range), inertias, 'bo-')
plt.xlabel('Broj grupa K')
plt.ylabel('J (inercija)')
plt.title(f'Elbow metoda - optimalni K (flagc={flagc})')
plt.xticks(list(K_range))
plt.tight_layout()
plt.show()