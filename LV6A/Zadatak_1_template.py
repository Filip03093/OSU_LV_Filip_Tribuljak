import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score

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
                    label=cl)


# ucitaj podatke
data = pd.read_csv("Social_Network_Ads.csv")
print(data.info())

data.hist()
plt.show()

# dataframe u numpy
X = data[["Age","EstimatedSalary"]].to_numpy()
y = data["Purchased"].to_numpy()

# podijeli podatke u omjeru 80-20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y, random_state = 10)

# skaliraj ulazne velicine
sc = StandardScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform((X_test))

# Model logisticke regresije
LogReg_model = LogisticRegression(penalty=None) 
LogReg_model.fit(X_train_n, y_train)

# Evaluacija modela logisticke regresije
y_train_p = LogReg_model.predict(X_train_n)
y_test_p = LogReg_model.predict(X_test_n)

print("Logisticka regresija: ")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p))))

# granica odluke pomocu logisticke regresije
plot_decision_regions(X_train_n, y_train, classifier=LogReg_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("Tocnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
plt.tight_layout()
plt.show()



KNN_model = KNeighborsClassifier(n_neighbors=5)
KNN_model.fit(X_train_n, y_train)

y_train_p_KNN = KNN_model.predict(X_train_n)
y_test_p_KNN = KNN_model.predict(X_test_n)

print("KNN (K=5):")
print("Tocnost train: " + "{:0.3f}".format(accuracy_score(y_train, y_train_p_KNN)))
print("Tocnost test:  " + "{:0.3f}".format(accuracy_score(y_test,  y_test_p_KNN)))

plot_decision_regions(X_train_n, y_train, classifier=KNN_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("KNN K=5, Tocnost train: " + "{:0.3f}".format(accuracy_score(y_train, y_train_p_KNN)))
plt.tight_layout()
plt.show()



for k in [1, 100]:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_n, y_train)

    y_tr = model.predict(X_train_n)
    y_te = model.predict(X_test_n)

    print(f"KNN (K={k}):")
    print("Tocnost train: " + "{:0.3f}".format(accuracy_score(y_train, y_tr)))
    print("Tocnost test:  " + "{:0.3f}".format(accuracy_score(y_test, y_te)))

    plot_decision_regions(X_train_n, y_train, classifier=model)
    plt.xlabel('x_1')
    plt.ylabel('x_2')
    plt.legend(loc='upper left')
    plt.title(f"KNN K={k} | Train: {accuracy_score(y_train, y_tr):.3f} | Test: {accuracy_score(y_test, y_te):.3f}")
    plt.tight_layout()
    plt.show()



k_values = range(1, 30)
cv_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_n, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

optimal_k = k_values[np.argmax(cv_scores)]
print("Optimalni K: " + str(optimal_k))
print("Tocnost CV:  " + "{:0.3f}".format(max(cv_scores)))

plt.figure()
plt.plot(k_values, cv_scores, marker='o')
plt.xlabel('K')
plt.ylabel('CV tocnost')
plt.title('Unakrsna validacija za KNN')
plt.xticks(k_values)
plt.tight_layout()
plt.show()



C_values = [0.1, 1, 10]
gamma_values = [0.1, 1, 10]

for C in C_values:
    for gamma in gamma_values:
        SVM_model = svm.SVC(kernel='rbf', C=C, gamma=gamma)
        SVM_model.fit(X_train_n, y_train)

        y_train_p_SVM = SVM_model.predict(X_train_n)
        y_test_p_SVM  = SVM_model.predict(X_test_n)

        print(f"SVM RBF C={C}, gamma={gamma}")
        print("Tocnost train: " + "{:0.3f}".format(accuracy_score(y_train, y_train_p_SVM)))
        print("Tocnost test:  " + "{:0.3f}".format(accuracy_score(y_test,  y_test_p_SVM)))

        plot_decision_regions(X_train_n, y_train, classifier=SVM_model)
        plt.xlabel('x_1')
        plt.ylabel('x_2')
        plt.legend(loc='upper left')
        plt.title(f"SVM RBF C={C}, gamma={gamma} | Test: {accuracy_score(y_test, y_test_p_SVM):.3f}")
        plt.tight_layout()
        plt.show()



kernels = ['linear', 'poly', 'rbf', 'sigmoid']

for kernel in kernels:
    SVM_model = svm.SVC(kernel=kernel)
    SVM_model.fit(X_train_n, y_train)

    y_train_p_SVM = SVM_model.predict(X_train_n)
    y_test_p_SVM  = SVM_model.predict(X_test_n)

    print(f"SVM kernel={kernel}")
    print("Tocnost train: " + "{:0.3f}".format(accuracy_score(y_train, y_train_p_SVM)))
    print("Tocnost test:  " + "{:0.3f}".format(accuracy_score(y_test,  y_test_p_SVM)))

    plot_decision_regions(X_train_n, y_train, classifier=SVM_model)
    plt.xlabel('x_1')
    plt.ylabel('x_2')
    plt.legend(loc='upper left')
    plt.title(f"SVM kernel={kernel} | Test: {accuracy_score(y_test, y_test_p_SVM):.3f}")
    plt.tight_layout()
    plt.show()



param_grid = {
    'model__C': [0.1, 1, 10, 100],
    'model__gamma': [10, 1, 0.1, 0.01]
}

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', svm.SVC(kernel='rbf'))
])

svm_gscv = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
svm_gscv.fit(X_train, y_train)

print("Optimalni parametri: " + str(svm_gscv.best_params_))
print("Tocnost CV: " + "{:0.3f}".format(svm_gscv.best_score_))

best_model = svm_gscv.best_estimator_
y_train_p_best = best_model.predict(X_train)
y_test_p_best  = best_model.predict(X_test)

print("Tocnost train: " + "{:0.3f}".format(accuracy_score(y_train, y_train_p_best)))
print("Tocnost test:  " + "{:0.3f}".format(accuracy_score(y_test, y_test_p_best)))

X_train_scaled = best_model.named_steps['scaler'].transform(X_train)
plot_decision_regions(X_train_scaled, y_train, classifier=best_model.named_steps['model'])
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("SVM optimal: " + str(svm_gscv.best_params_))
plt.tight_layout()
plt.show()