# 导入数据集
from sklearn.datasets import load_iris
iris = load_iris()
raw_x = iris.data
raw_y = iris.target

# 实验1：探究训练集与测试集的划分比例对准确率的影响
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from collections import  Counter
import matplotlib.pyplot as plt

index = []
acc = []
for i in range(1,6):
    index.append(0.1 * i)
    x_train, x_test, y_train, y_test = train_test_split(raw_x, raw_y, test_size=0.1*i, random_state=0)
    model = KNeighborsClassifier(n_neighbors=10)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    acc.append(Counter(y_pred==y_test)[True] / x_test.shape[0])

plt.plot(index, acc)
plt.show()