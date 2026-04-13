import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt


model = keras.models.load_model("FCN.keras")
print("Model uspješno učitan!")

(_, _), (x_test, y_test) = keras.datasets.mnist.load_data()


x_test_s = x_test.astype("float32") / 255
x_test_s = np.expand_dims(x_test_s, -1)

num_classes = 10
y_test_s = keras.utils.to_categorical(y_test, num_classes)


predictions = model.predict(x_test_s)

y_pred = np.argmax(predictions, axis=1)
y_true = np.argmax(y_test_s, axis=1)


misclassified_indices = np.where(y_pred != y_true)[0]

print(f"Ukupno loše klasificiranih: {len(misclassified_indices)} / {len(y_true)}")


num_to_show = 10

fig, axes = plt.subplots(2, 5, figsize=(14, 6))
axes = axes.flatten()

for i, idx in enumerate(misclassified_indices[:num_to_show]):
    axes[i].imshow(x_test[idx], cmap='gray')
    axes[i].set_title(
        f'Stvarna: {y_true[idx]}\nPredviđena: {y_pred[idx]}',
        fontsize=10,
        color='red'
    )
    axes[i].axis('off')

plt.suptitle('Loše klasificirane slike — testni skup', fontsize=14)
plt.tight_layout()
plt.show()