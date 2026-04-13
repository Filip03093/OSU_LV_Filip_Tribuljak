import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt
from PIL import Image


model = keras.models.load_model("FCN.keras")
print("Model uspješno učitan!")

img = Image.open("test.png")

plt.figure(figsize=(3, 3))
plt.imshow(img)
plt.title("Originalna slika (test.png)")
plt.axis('off')
plt.show()


img = img.convert("L")

img = img.resize((28, 28), Image.LANCZOS)

img_array = np.array(img).astype("float32")

img_array = 255.0 - img_array

img_array = img_array / 255.0

plt.figure(figsize=(3, 3))
plt.imshow(img_array, cmap='gray')
plt.title("Slika prilagođena za mrežu (28x28)")
plt.axis('off')
plt.show()

img_array = np.expand_dims(img_array, axis=-1)
img_array = np.expand_dims(img_array, axis=0)

print(f"Oblik ulaza u mrežu: {img_array.shape}")


predictions = model.predict(img_array)

print("\nVjerojatnosti po klasama:")
for digit, prob in enumerate(predictions[0]):
    print(f"  Znamenka {digit}: {prob*100:.2f}%")

predicted_class = np.argmax(predictions[0])
confidence = np.max(predictions[0]) * 100

print(f"\nPredviđena znamenka: {predicted_class}")
print(f"Pouzdanost: {confidence:.2f}%")

plt.figure(figsize=(8, 4))
plt.bar(range(10), predictions[0] * 100, color='steelblue')
plt.xticks(range(10))
plt.xlabel('Znamenka')
plt.ylabel('Vjerojatnost (%)')
plt.title(f'Predikcija mreže — Predviđena znamenka: {predicted_class} ({confidence:.1f}%)')
plt.tight_layout()
plt.show()