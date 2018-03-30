# -*- coding: utf-8 -*-
# @Time    : 2018/3/22 17:04
# @Author  : timothy

import pandas as pd
from sklearn import cross_validation, metrics
from sklearn import tree
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals.six import StringIO
import pydot
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn import pipeline
from sklearn.metrics import classification_report
from sklearn.grid_search import GridSearchCV


# 读取数据集
def read_data():
    file_path = 'resources/telecom_train.csv'
    df = pd.read_csv(file_path)
    attribute_list = ['gender', 'edu_class', 'feton', 'prom', 'posPlanChange', 'curPlan', 'call_10086']
    X = df.loc[:, attribute_list]
    y = df.loc[:, 'churn']
    # show_data(X.loc[:, 'call_10086'])
    print(df['churn'].value_counts())
    return X, y


# 归一化
def scale(X):
    X_np = np.array(X.values)
    X_scaled = preprocessing.robust_scale(X_np)
    return X_scaled


# 用交叉验证法，划分训练集和测试集
def train_test_par(X, y):
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)  # 百分之20测试
    return X_train, X_test, y_train, y_test


# 数据可视化
def show_data(X):
    a = np.array(X.values)
    plt.bar(np.arange(1000), a[0:1000])
    plt.show()


# 主成分分析降维
def pca_decomposition(X):
    pca = PCA(n_components=5)  # 原始7维降为5维
    pca.fit(X)
    print(sum(pca.explained_variance_ratio_))  # 5维特征方差和占比，100%说明降维没损失信息
    return X


# 普通决策树
def use_dtree(X_train, X_test, y_train, y_test):
    clf = tree.ExtraTreeClassifier()
    clf = clf.fit(X_train, y_train)
    expected = y_test
    predicted = clf.predict(X_test)
    print("Classification report for classifier %s:\n%s\n"
          % (clf, metrics.classification_report(expected, predicted)))
    return clf


# 搜索决策树最优参数
def grid_search(X_train, X_test, y_train, y_test):
    pl = pipeline.Pipeline([
        ('clf', RandomForestClassifier(criterion='gini'))
    ])
    # 设置要尝试的参数组合
    parameters = {
        'clf__n_estimators': (5, 10, 20, 50),
        'clf__max_depth': (30, 50, 150),
        'clf__min_samples_split': (8, 9, 10),
        'clf__min_samples_leaf': (5, 7, 9)
    }
    grid_search = GridSearchCV(pl, parameters, n_jobs=-1, verbose=1, scoring='f1')  # n_job=-1表示cpu多少个核就启动多少个并行
    grid_search.fit(X_train, y_train)
    print('最佳效果：%0.3f' % grid_search.best_score_)
    print('最优参数：')
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print('\t%s: %r' % (param_name, best_parameters[param_name]))
    predictions = grid_search.predict(X_test)
    print(classification_report(y_test, predictions))


def draw_tree(clf):
    '''
        画决策树
    '''
    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph[0].write_png('result/telecom.png')


def main():
    X, y = read_data()
    # pca_decomposition(X)
    # print('归一化前')
    X_train, X_test, y_train, y_test = train_test_par(X, y)
    grid_search(X_train, X_test, y_train, y_test)

    # # 归一化
    # X = scale(X)
    #
    # print('归一化后')
    # X_train, X_test, y_train, y_test = train_test_par(X,y)
    # clf = use_dtree(X_train, X_test, y_train, y_test)
    # draw_tree(clf)


if __name__ == '__main__':
    main()
