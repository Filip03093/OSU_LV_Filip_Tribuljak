import numpy as np
import matplotlib.pyplot as plt

img = plt.imread("road.jpg")
img = img[:, :, 0].copy()

img_bright = np.clip(img.astype(np.int32) + 80, 0, 255).astype(np.uint8)

w = img.shape[1]
img_quarter = img[:, w//4 : w//2]

img_rotated = np.rot90(img, k=-1)

img_flip_h = np.fliplr(img)
img_flip_v = np.flipud(img)

fig, axes = plt.subplots(2, 3, figsize=(12, 7))

axes[0, 0].imshow(img, cmap='gray'); axes[0, 0].set_title('Original')
axes[0, 1].imshow(img_bright, cmap='gray'); axes[0, 1].set_title('a) Brighter')
axes[0, 2].imshow(img_quarter, cmap='gray'); axes[0, 2].set_title('b) Second quarter')
axes[1, 0].imshow(img_rotated, cmap='gray'); axes[1, 0].set_title('c) Rotation 90° ↻')
axes[1, 1].imshow(img_flip_h, cmap='gray'); axes[1, 1].set_title('d) Mirrored ↔')
axes[1, 2].imshow(img_flip_v, cmap='gray'); axes[1, 2].set_title('d) Mirrored ↕')

for ax in axes.flat:
    ax.axis('off')

plt.tight_layout()
plt.show()
