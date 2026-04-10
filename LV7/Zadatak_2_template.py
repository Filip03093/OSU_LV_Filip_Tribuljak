import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
from sklearn.cluster import KMeans

# ucitaj sliku
for img_idx in range(1,7):
    img_path = f"imgs\\test_{img_idx}.jpg"
    img = Image.imread(img_path)

    # prikazi originalnu sliku
    plt.figure()
    plt.title("Originalna slika")
    plt.imshow(img)
    plt.tight_layout()
    plt.show()

    # pretvori vrijednosti elemenata slike u raspon 0 do 1
    img = img.astype(np.float64) / 255

    # transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
    w,h,d = img.shape
    img_array = np.reshape(img, (w*h, d))

    # rezultatna slika
    img_array_aprox = img_array.copy()



    unique_colors = np.unique(img_array, axis=0)
    print(f"Broj razlicitih boja u slici: {len(unique_colors)}")



    fig, axes = plt.subplots(2, 5, figsize=(18, 7))

    for i, K in enumerate(range(1, 6)):

        km = KMeans(n_clusters=K, n_init=5, random_state=0)
        km.fit(img_array)

        labels = km.labels_
        centers = km.cluster_centers_

        img_array_aprox = centers[labels]
        img_aprox = np.reshape(img_array_aprox, (w, h, d))

        axes[0, i].imshow(img)
        axes[0, i].set_title(f'Originalna')
        axes[0, i].axis('off')

        axes[1, i].imshow(np.clip(img_aprox, 0, 1))
        axes[1, i].set_title(f'K = {K}')
        axes[1, i].axis('off')

    plt.suptitle('Usporedba originalne i kvantizirane slike (K = 1 do 5)', fontsize=14)
    plt.tight_layout()
    plt.show()



    inertias = []
    K_range = range(1, 11)

    for k in K_range:
        km_test = KMeans(n_clusters=k, n_init=3, random_state=0)
        km_test.fit(img_array)
        inertias.append(km_test.inertia_)

    plt.figure()
    plt.plot(list(K_range), inertias, 'bo-')
    plt.xlabel('Broj grupa K')
    plt.ylabel('J (inercija)')
    plt.title('Elbow metoda – kvantizacija slike')
    plt.xticks(list(K_range))
    plt.tight_layout()
    plt.show()



    fig, axes = plt.subplots(1, K, figsize=(3 * K, 3))

    for k in range(K):
        binary_mask = (labels == k).reshape(w, h)
        axes[k].imshow(binary_mask, cmap='gray')
        axes[k].set_title(f'Grupa {k}')
        axes[k].axis('off')

    plt.suptitle('Binarne slike po grupama')
    plt.tight_layout()
    plt.show()