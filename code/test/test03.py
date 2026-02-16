from sklearn.datasets import load_iris
iris = load_iris()
raw_x = iris.data
raw_y = iris.target
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np

# 实验3：探究是否对样本进行归一化对准确率的影响
index = [] #保存交叉验证的序号
acc_list = [] #保存交叉验证的准确率
k = 5
model = KNeighborsClassifier(n_neighbors=9,weights='distance',p=3)

from sklearn.preprocessing import StandardScaler # 导入归一化的包
ss = StandardScaler()
raw_x = ss.fit_transform(raw_x)

from sklearn.model_selection import cross_val_score # k折交叉验证
acc = cross_val_score(model,raw_x,raw_y,cv=k,scoring='accuracy')

index = list(range(1,k+1,1))
acc_list = acc
plt.plot(np.array(index),np.array(acc_list))

plt.show()

print(np.array(acc_list).mean())