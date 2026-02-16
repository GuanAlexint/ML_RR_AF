# 导入数据集
from sklearn.datasets import load_iris
iris = load_iris()
raw_x = iris.data
raw_y = iris.target

# 实验 2：探究模型的泛化能力
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from collections import  Counter
import matplotlib.pyplot as plt
import numpy as np

index = [] #保存交叉验证的序号
acc_list = [] #保存交叉验证的准确率
k = 10 #十折交叉验证
for i in range(k):
    # 划分训练集和测试集
    num = raw_x.shape[0] #总共的样本数据
    index_start = int(i*num/k)
    index_end = int((i+1)*num/k)
    x_test = raw_x[index_start:index_end,:]
    x_train1 = raw_x[0:index_start, :]
    x_train2 = raw_x[index_end:, :]
    x_train = np.concatenate([x_train1,x_train2]) #把两端训练集拼接起来
    #相同方法划分标签
    y_test = raw_y[index_start:index_end]
    y_train1 = raw_y[0:index_start]
    y_train2 = raw_y[index_end:]
    y_train = np.concatenate([y_train1, y_train2])

    #构建KNN模型，k=9，距离采用范数为3的闵可夫斯基距离
    model = KNeighborsClassifier(n_neighbors=9,weights='distance',p=3)
    model.fit(x_train,y_train) #模型训练
    y_pred = model.predict(x_test) #预测测试集

    #计算准确率
    acc = Counter(y_pred==y_test)[True] / x_test.shape[0]

    #记录实验结果
    index.append(i+1)
    acc_list.append(acc)

#绘图：折数-准确率 曲线
plt.plot(np.array(index),np.array(acc_list))
# plt.show()
print(np.array(acc_list).mean())

