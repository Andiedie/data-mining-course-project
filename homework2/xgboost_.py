import random
import pandas as pd
import xgboost as xgb

dtest = xgb.DMatrix('./dataset/test.txt')
dtrain_ = xgb.DMatrix('./dataset/train.txt')

rows = dtrain_.num_row()
validation_rate = 0.08
indices = list(range(0, rows))
random.shuffle(indices)
bound_index = int(rows * validation_rate)
validation_index = indices[:bound_index]
train_index = indices[bound_index:]
dvalidate = dtrain_.slice(validation_index)
dtrain = dtrain_.slice(train_index)

param = {'objective': 'binary:logistic'}

bst = xgb.train(param, dtrain, num_boost_round=10000, evals=[
                (dvalidate, 'eval')], early_stopping_rounds=100)

bst = xgb.train(param, dtrain_, num_boost_round=bst.best_iteration)

result = bst.predict(dtest)
pd.Series(result).to_csv(
    path='./output/xgboost.csv',
    header=['label'],
    index_label='id'
)
