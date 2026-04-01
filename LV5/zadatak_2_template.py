import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report

labels= {0:'Adelie', 1:'Chinstrap', 2:'Gentoo'}

def plot_decision_regions(X, y, classifier, resolution=0.02):
    plt.figure()
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    edgecolor = 'w',
                    label=labels[cl])

# ucitaj podatke
df = pd.read_csv("penguins.csv")

# izostale vrijednosti po stupcima
print(df.isnull().sum())

# spol ima 11 izostalih vrijednosti; izbacit cemo ovaj stupac
df = df.drop(columns=['sex'])

# obrisi redove s izostalim vrijednostima
df.dropna(axis=0, inplace=True)

# kategoricka varijabla vrsta - kodiranje
df['species'].replace({'Adelie' : 0,
                        'Chinstrap' : 1,
                        'Gentoo': 2}, inplace = True)

print(df.info())

# izlazna velicina: species
output_variable = ['species']

# ulazne velicine: bill length, flipper_length
input_variables = ['bill_length_mm',
                    'flipper_length_mm']

X = df[input_variables].to_numpy()
y = df['species'].to_numpy()
# podjela train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)


classes, train_counts = np.unique(y_train, return_counts=True)
_, test_counts = np.unique(y_test, return_counts=True)
x = np.arange(len(classes))
plt.bar(x - 0.2, train_counts, width=0.4, label='train')
plt.bar(x + 0.2, test_counts,  width=0.4, label='test')
plt.xticks(x, [labels[c] for c in classes])
plt.xlabel('Vrsta'); plt.ylabel('Broj primjera'); plt.legend(); plt.show()


model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


print("intercept:", model.intercept_)
print("coef:\n", model.coef_)


plot_decision_regions(X_train, y_train, model)
plt.xlabel('bill_length_mm')
plt.ylabel('flipper_length_mm')
plt.legend()
plt.show()


y_pred = model.predict(X_test)
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=list(labels.values())).plot()
plt.show()
print("Točnost:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=list(labels.values())))


input_variables_all = ['bill_length_mm', 'flipper_length_mm', 'bill_depth_mm', 'body_mass_g']
X_all = df[input_variables_all].to_numpy()
X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(X_all, y, test_size=0.2, random_state=123)

model_all = LogisticRegression(max_iter=1000)
model_all.fit(X_train_all, y_train_all)
y_pred_all = model_all.predict(X_test_all)

ConfusionMatrixDisplay(confusion_matrix(y_test_all, y_pred_all),
                       display_labels=list(labels.values())).plot()
plt.show()
print("Točnost (više značajki):", accuracy_score(y_test_all, y_pred_all))
print(classification_report(y_test_all, y_pred_all, target_names=list(labels.values())))
