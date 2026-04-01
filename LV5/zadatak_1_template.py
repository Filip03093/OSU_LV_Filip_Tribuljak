import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score

X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                           random_state=213, n_clusters_per_class=1, class_sep=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

plt.scatter(X_train[:,0], X_train[:,1], c=y_train, cmap='bwr', label='train')
plt.scatter(X_test[:,0], X_test[:,1], c=y_test, cmap='bwr', marker='x', label='test')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.show()

model = LogisticRegression()
model.fit(X_train, y_train)

t0, (t1, t2) = model.intercept_[0], model.coef_[0]
x1 = np.linspace(X[:,0].min(), X[:,0].max(), 100)
x2 = -(t0 + t1*x1) / t2
plt.scatter(X_train[:,0], X_train[:,1], c=y_train, cmap='bwr')
plt.plot(x1, x2, 'g-')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot()
plt.show()
print("Točnost:", accuracy_score(y_test, y_pred))
print("Preciznost:", precision_score(y_test, y_pred))
print("Odziv:", recall_score(y_test, y_pred))

correct = y_pred == y_test
plt.scatter(X_test[correct,0], X_test[correct,1], color='green')
plt.scatter(X_test[~correct,0], X_test[~correct,1], color='black', marker='x')
plt.xlabel('x1') 
plt.ylabel('x2') 
plt.show()
