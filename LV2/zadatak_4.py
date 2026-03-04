import numpy as np
import matplotlib.pyplot as plt

black = np.zeros((50, 50), dtype=np.uint8)
white = np.ones((50, 50), dtype=np.uint8) * 255

upper_row = np.hstack([black,  white])
lower_row = np.hstack([white, black])

image = np.vstack([upper_row, lower_row])

print(f"image shape: {image.shape}")

plt.imshow(image, cmap='gray', vmin=0, vmax=255)
plt.show()
