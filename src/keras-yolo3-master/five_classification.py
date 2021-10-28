from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.preprocessing import label_binarize
import numpy as np
from sklearn.multiclass import OneVsRestClassifier


def train(x, y):
    print(x)
    # y = label_binarize(y, classes=[0, 1, 2, 3, 4])
    print(y)
    # 设置种类
    # n_classes = y.shape[1]
    # print(n_classes)
    # 训练模型并预测
    random_state = np.random.RandomState(0)
    n_samples, n_features = x.shape

    # 随机化数据，并划分训练数据和测试数据
    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)
    model = svm.LinearSVC()
    clt = model.fit(X_train, y_train)
    joblib.dump(model, 'logs/model.pkl')
    print("train accuracy:", model.score(X_test, y_test))
