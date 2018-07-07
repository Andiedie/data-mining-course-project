import os
import pickle
import timeit
import logging
import argparse
import threading
import numpy as np
import pandas as pd
from sklearn.datasets import load_svmlight_file
from sklearn.utils import shuffle
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

LOG_FORMAT = "[%(asctime)s] [%(levelname)s] - %(message)s"
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)

parser = argparse.ArgumentParser()
parser.add_argument('--algorithm', '-algo', default=None,
                    type=list, help='0 <= algorithm <= 9')
argvs = parser.parse_args()

classifiers = [
    KNeighborsClassifier(3),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(n_estimators=180, learning_rate=0.9),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
    SVC(kernel="linear", C=0.025, probability=True),
    SVC(gamma=2, C=1, probability=True)
]
names = [
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
    argvs.algorithm = list(map(int, argvs.algorithm))
    names = [names[index] for index in argvs.algorithm]
    classifiers = [classifiers[index] for index in argvs.algorithm]
logging.info('classifiers ready: %s' % names)

data = load_svmlight_file('./dataset/train.txt', n_features=201)
logging.info('train data loaded')
x_train, y_train = shuffle(data[0], data[1])
x_train = x_train.toarray()
logging.info('train data shuffled')
data = load_svmlight_file('./dataset/test.txt', n_features=201)
x_predict = data[0].toarray()
logging.info('predict data loaded')

for name, clf in zip(names, classifiers):
    start = timeit.default_timer()
    model_file = os.path.join('./model', name + '.model')
    logging.info('%s fitting' % name)
    clf.fit(x_train, y_train)
    logging.info('%s fitted' % name)
    pickle.dump(clf, open(model_file, 'wb'))
    logging.info('%s saved' % name)
    output_file = os.path.join('./output', name + '.csv')
    logging.info('%s predicting' % name)
    predict = clf.predict_proba(x_predict)
    predict = np.array(predict)
    pd.Series(predict[:, 1]).to_csv(
        path=output_file,
        header=['label'],
        index_label='id'
    )
    logging.info('%s predict done' % name)
    end = timeit.default_timer()
    logging.info('%s takes %d seconds' % (name, end - start))
