from sklearn.datasets import load_iris
iris = load_iris()
raw_x = iris.data
raw_y = iris.target
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np

acc_list = []
k = 5
m = 20
index = np.array(range(1,m+1,1))
from sklearn.model_selection import cross_val_score
for i in range(1,m+1):
    model = KNeighborsClassifier(n_neighbors=i,weights='distance',p=3)
    acc = cross_val_score(model,raw_x,raw_y,cv=k,scoring='accuracy')
    mean_acc = np.array(acc).mean()
    print(mean_acc)
    acc_list.append(mean_acc)

plt.plot(index, np.array(acc_list))
plt.xlim(0,m+1)
plt.show()