import os
import timeit
import pickle
import logging
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.datasets import load_svmlight_file
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV

LOG_FORMAT = "[%(asctime)s] [%(levelname)s] - %(message)s"
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)

data = load_svmlight_file('./dataset/train.txt', n_features=201)
logging.info('train data loaded')
x_train, y_train = shuffle(data[0], data[1])
x_train = x_train.toarray()
logging.info('train data shuffled')
data = load_svmlight_file('./dataset/test.txt', n_features=201)
x_predict = data[0].toarray()
logging.info('predict data loaded')

start = timeit.default_timer()
clf = AdaBoostClassifier()
test_case = {
  'n_estimators': range(180, 300, 10),
  'learning_rate': [
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
  ]
}
search = GridSearchCV(
    clf,
    param_grid=test_case,
    scoring='roc_auc',
    n_jobs=10
)
search.fit(x_train, y_train)
print('----------best_params----------')
print(search.best_params_)
print('----------best_score----------')
print(search.best_score_)
end = timeit.default_timer()
logging.info('search takes %d seconds' % (end - start))
model_file = './model/AdaBoost.model'
clf = AdaBoostClassifier(
    n_estimators=search.best_params_['n_estimators'],
    learning_rate=search.best_params_['learning_rate']
)
logging.info('AdaBoost fitting')
clf.fit(x_train, y_train)
logging.info('AdaBoost fitted')
pickle.dump(clf, open(model_file, 'wb'))
logging.info('AdaBoost saved')
output_file = './output/AdaBoost.csv'
logging.info('AdaBoost predicting')
predict = clf.predict_proba(x_predict)
predict = np.array(predict)
pd.Series(predict[:, 1]).to_csv(
    path=output_file,
    header=['label'],
    index_label='id'
)
logging.info('AdaBoost predict done')
