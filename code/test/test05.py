from sklearn import svm
from sklearn import datasets

# 首先简单建立与训练一个SVCModel
clf = svm.SVC()
iris = datasets.load_iris()
X,y = iris.data,iris.target
clf.fit(X,y)

# 使用pickle来保存与读取训练好的Model
import pickle
with open('../save/clf.pickle', 'wb')as f:
    pickle.dump(clf,f)

with open('../save/clf.pickle', 'rb')as f:
    clf2 = pickle.load(f)
    print(clf2.predict(X[0:1]))


# 使用joblib 保存
import joblib
joblib.dump(clf, '../save/clf.pkl')
clf3 = joblib.load('../save/clf.pkl')
print(clf3.predict(X[0:1]))
