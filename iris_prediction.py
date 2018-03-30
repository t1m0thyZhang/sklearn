# -*- coding: utf-8 -*-
# @Time    : 2018/3/22 17:18
# @Author  : timothy

from sklearn.datasets import load_iris
from sklearn import tree, metrics
from sklearn import cross_validation
from sklearn.externals.six import StringIO
import pydot

iris = load_iris()
X = iris.data
y = iris.target
# 用交叉验证法，划分训练集和测试集
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)  # 百分之20测试
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
expected = y_test
predicted = clf.predict(X_test)
print("Classification report for classifier %s:\n%s\n"
      % (clf, metrics.classification_report(expected, predicted)))


# 把生成的决策树画出来
dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph[0].write_png('iris_simple.png')

