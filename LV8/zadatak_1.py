import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# train i test podaci
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# prikaz karakteristika train i test podataka
print('Train: X=%s, y=%s' % (x_train.shape, y_train.shape))
print('Test: X=%s, y=%s' % (x_test.shape, y_test.shape))

# TODO: prikazi nekoliko slika iz train skupa
index = 0
plt.figure()
plt.imshow(x_train[index], cmap='gray')
plt.title(f'Oznaka: {y_train[index]}')
plt.axis('off')
plt.show()
print(f'Oznaka slike na indeksu {index}: {y_train[index]}')

# skaliranje slike na raspon [0,1]
x_train_s = x_train.astype("float32") / 255
x_test_s = x_test.astype("float32") / 255

# slike trebaju biti (28, 28, 1)
x_train_s = np.expand_dims(x_train_s, -1)
x_test_s = np.expand_dims(x_test_s, -1)

print("x_train shape:", x_train_s.shape)
print(x_train_s.shape[0], "train samples")
print(x_test_s.shape[0], "test samples")


# pretvori labele
y_train_s = keras.utils.to_categorical(y_train, num_classes)
y_test_s = keras.utils.to_categorical(y_test, num_classes)


# TODO: kreiraj model pomocu keras.Sequential(); prikazi njegovu strukturu
model = keras.Sequential()
model.add(layers.Input(shape=(28, 28, 1)))
model.add(layers.Flatten())
model.add(layers.Dense(100, activation="relu"))
model.add(layers.Dense(50, activation="relu"))
model.add(layers.Dense(num_classes, activation="softmax"))

model.summary()


# TODO: definiraj karakteristike procesa ucenja pomocu .compile()
model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)


# TODO: provedi ucenje mreze
batch_size = 32
epochs     = 15

history = model.fit(
    x_train_s,
    y_train_s,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.1
)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history.history['accuracy'], label='Train accuracy')
ax1.plot(history.history['val_accuracy'], label='Val accuracy')
ax1.set_title('Točnost po epohama')
ax1.set_xlabel('Epoha')
ax1.set_ylabel('Točnost')
ax1.legend()

ax2.plot(history.history['loss'], label='Train loss')
ax2.plot(history.history['val_loss'], label='Val loss')
ax2.set_title('Gubitak po epohama')
ax2.set_xlabel('Epoha')
ax2.set_ylabel('Gubitak')
ax2.legend()

plt.tight_layout()
plt.show()


# TODO: Prikazi test accuracy i matricu zabune
score = model.evaluate(x_test_s, y_test_s, verbose=0)
print(f"\nTest loss: {score[0]:.4f}")
print(f"Test accuracy: {score[1]:.4f}")


predictions = model.predict(x_test_s)

y_pred = np.argmax(predictions, axis=1)
y_true = np.argmax(y_test_s, axis=1)

cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(10)))
fig, ax = plt.subplots(figsize=(9, 9))
disp.plot(ax=ax)
ax.set_title('Matrica zabune — testni skup')
plt.tight_layout()
plt.show()

# TODO: spremi model
model.save("FCN.keras")
print("Model pohranjen")
