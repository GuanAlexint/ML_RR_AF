# 导入数据集
from sklearn.datasets import load_iris
iris = load_iris()
raw_x = iris.data
raw_y = iris.target

# 实验 2：探究模型的泛化能力
# from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from collections import  Counter
import matplotlib.pyplot as plt
import numpy as np

index = [] #保存交叉验证的序号
acc_list = [] #保存交叉验证的准确率
k = 5
model = KNeighborsClassifier(n_neighbors=9,weights='distance',p=3)

from sklearn.model_selection import cross_val_score # k折交叉验证
acc = cross_val_score(model,raw_x,raw_y,cv=k,scoring='accuracy')

index = list(range(1,k+1,1))
acc_list = acc
plt.plot(np.array(index),np.array(acc_list))

plt.show()

print(np.array(acc_list).mean())