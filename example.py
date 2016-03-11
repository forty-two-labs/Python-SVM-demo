import numpy as np
import csv
from sklearn import svm, metrics, preprocessing
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold

with open('data.csv', 'rb') as f:  
    users = np.array(list(csv.reader(f)), dtype=np.float64)

split_at = int(round(len(users) * 0.8))

X_train = users[:split_at,:-1]
y_train = users[:split_at,-1]
X_test = users[split_at:,:-1]
y_test = users[split_at:,-1]

scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


param_grid = {"C": np.logspace(-2, 5, 8), "kernel": ["rbf"]}
cv = StratifiedKFold(y_train, n_folds=2)
grid = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=cv)
grid.fit(X_train, y_train)

y_pred = grid.predict(X_test)

print(metrics.confusion_matrix(y_test, y_pred))
print(np.sum(y_pred == y_test) / float(len(y_pred)))