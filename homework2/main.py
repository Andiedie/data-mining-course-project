import os
import sys
import pickle
import timeit
import argparse
import numpy as np
import pandas as pd
from sklearn.datasets import load_svmlight_file
from sklearn.utils import shuffle
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

parser = argparse.ArgumentParser()
parser.add_argument('--algorithm', '-algo', default=None, type=int, help='0 <= algorithm <= 9')
argvs = parser.parse_args(sys.argv[1:])

classifiers = [
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    KNeighborsClassifier(3),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1)
]
names = [
    "Gaussian Process",
    "Nearest Neighbors",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
    "Naive Bayes",
    "QDA",
    "Linear SVM",
    "RBF SVM"
]
if argvs.algorithm is not None:
    names = [names[argvs.algorithm]]
    classifiers = [classifiers[argvs.algorithm]]
print('classifiers ready', names)

data = load_svmlight_file('./dataset/train.txt', n_features=201)
print('train loaded')
x_train, y_train = shuffle(data[0], data[1])
x_train = x_train.toarray()
print('train shuffled')
data = load_svmlight_file('./dataset/test.txt', n_features=201)
x_predict = data[0].toarray()
print('predict loaded')

for name, clf in zip(names, classifiers):
    start = timeit.default_timer()
    model_file = os.path.join('./model', name + '.model')
    print(name, 'fitting')
    clf.fit(x_train.toarray(), y_train)
    print(name, 'fitted')
    pickle.dump(clf, open(model_file, 'wb'))
    print(name, 'saved')
    output_file = os.path.join('./output', name + '.model')
    print(name, 'predicting')
    predict = clf.predict_proba(x_predict)
    predict = np.array(predict)
    pd.Series(predict[:,1]).to_csv(
        path=output_file,
        header=['label'],
        index_label='id'
    )
    print(name, 'predict done')
    end = timeit.default_timer()
    print(name, 'takes', end - start, 'second')
