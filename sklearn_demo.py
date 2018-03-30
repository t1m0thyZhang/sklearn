# -*- coding: utf-8 -*-
# @Time    : 2018/3/11 15:13
# @Author  : timothy

import pandas as pd
from sklearn import cross_validation
from sklearn.svm import SVC
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

# 读取数据集
file_path = 'D:/tensorflow/resources/telecom_train.csv'
df = pd.read_csv(file_path)
attribute_list = ['gender', 'edu_class', 'feton', 'prom', 'posPlanChange', 'curPlan', 'call_10086']
# attribute_list = ['call_10086']
X = df.loc[:, attribute_list]
y = df.loc[:, 'churn']

clf_1_scores = []
clf_2_scores = []
clf_3_scores = []
for i in range(30):
    # 用交叉验证法，划分训练集和测试集
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)  # 百分之20测试

    # 决策树
    clf_1 = tree.DecisionTreeClassifier()
    clf_1 = clf_1.fit(X_train, y_train)
    score_1 = accuracy_score(y_test, clf_1.predict(X_test))  # 查看分类准确率
    print('决策树准确度:', score_1)
    clf_1_scores.append(score_1)

    # svm
    clf_2 = SVC(kernel='rbf')  # 选择svm内核
    clf_2.fit(X_train, y_train)  # 用svm进行分类
    score_2 = accuracy_score(y_test, clf_2.predict(X_test))  # 查看分类准确率
    print('svm准确度:', score_2)
    clf_2_scores.append(score_2)

    # 神经网络
    clf_3 = MLPClassifier(solver='adam', alpha=1e-4, hidden_layer_sizes=(4), random_state=1)
    clf_3 = clf_3.fit(X_train, y_train)
    score_3 = accuracy_score(y_test, clf_3.predict(X_test))  # 查看分类准确率
    print('神经网络准确度:', score_3)
    clf_3_scores.append(score_3)

clf_1_scores = sorted(clf_1_scores)
clf_2_scores = sorted(clf_2_scores)
clf_3_scores = sorted(clf_3_scores)
plt.plot(clf_1_scores, color='r', label='dtree')
plt.plot(clf_2_scores, color='g', label='svm')
plt.plot(clf_3_scores, color='b', label='nn')
plt.legend(loc='upper left')
plt.show()