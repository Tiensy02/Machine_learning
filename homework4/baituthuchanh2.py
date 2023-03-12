from sklearn.datasets import fetch_20newsgroups_vectorized
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
n_samples = 10000
X, y = fetch_20newsgroups_vectorized(subset='all', return_X_y=True)
X = X[:n_samples]
y = y[:n_samples]
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=42,stratify=y, test_size=0.1)
train_samples, n_features = X_train.shape
n_classes = np.unique(y).shape[0]
lorg=LogisticRegression(multi_class='multinomial',solver='sag', max_iter=5000)
# and train model by Training Dataset
lorg.fit(X_train,y_train)
# Then Predict the Test data
y_pred=lorg.predict(X_test)
# for accuracy
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))
# for confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)