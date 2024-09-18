

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pathlib import Path
np.random.seed(42)


#%%
class KNN:

    def __init__(self, K):
        self.K = K

    def distance(self, x1, x2):
        dist = 0
        for i in range(len(x1)):
            dist += (x1[i]-x2[i])**2
        return np.sqrt(dist)

    def knn_search(self, X_train, Y_train, Q):
        y_pred = np.zeros(Q.shape[0])

        for i, query in enumerate(Q):
            idx = np.argsort([self.distance(query, x) for x in X_train])[:self.K]
            knn_labels = Y_train[idx]
            y_pred[i] = np.bincount(knn_labels).argmax()

        return y_pred


#%%
if __name__ == '__main__':

    data_path = Path.home().joinpath('Documents', 'Data')
    iris = fetch_openml(name='iris', version=1, data_home=data_path)
    X = iris.data.iloc[:,:2].to_numpy()
    label2number = LabelEncoder()
    Y = label2number.fit_transform(iris.target)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=32)

    K = 5
    knn = KNN(K)
    Y_pred = knn.knn_search(X_train, Y_train, X_test)

    print(f"\nTrue labels in test dataset: {Y_test}")
    print(f"\nPredicted labels in test dataset: {Y_pred}")

    acc = np.sum(Y_test==Y_pred)/Y_test.shape[0]
    print(f"\nAccuracy: {acc}")


