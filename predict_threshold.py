# -*- coding: utf-8 -*-
# @Time    : 2018/3/22 10:02
# @Author  : timothy
'''
    根据网络和gamma值，预测节点级SIS模型灭绝阈值
'''

import numpy as np
import pandas as pd
from sklearn import cross_validation, linear_model, metrics
import matplotlib.pyplot as plt


def read_data():
    '''
        读取数据集
    :return: 输入值和标签
    '''
    file_path = 'resources/threshold2.csv'
    df = pd.read_csv(file_path)
    # attribute_list = ['node_num', 'edge_num', 'gamma']
    attribute_list = ['gamma']
    X = df.loc[:, attribute_list]
    y = df.loc[:, 'beta']
    show_data(X, y)
    # 用交叉验证法，划分训练集和测试集
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)  # 百分之20测试
    return X_train, X_test, y_train, y_test


def show_data(X_train, y_train):
    X = X_train.loc[:, 'gamma'].values
    Y = y_train.values
    plt.plot(X, Y, 'go')
    plt.xlabel('gamma')
    plt.ylabel('beta')
    plt.show()


def try_different_model(clf, X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)
    expected = y_test
    predicted = clf.predict(X_test)
    print("Classification report for classifier %s:\n%s\n"
          % (clf, metrics.classification_report(expected, predicted)))
    result = clf.predict(X_test)
    plt.plot(np.arange(50), result[0:50], 'go', label='predict')
    plt.plot(np.arange(50), y_test[0:50], 'ro', label='true')
    plt.show()


def main():
    X_train, X_test, y_train, y_test = read_data()
    clf = linear_model.LinearRegression()
    # print(X_train)
    # print(X_test)
    # try_different_model(clf, X_train, X_test, y_train, y_test)

if __name__ == '__main__':
    main()

