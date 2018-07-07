import random
import pandas as pd
import xgboost as xgb

dtest = xgb.DMatrix('./dataset/test.txt')
dtrain = xgb.DMatrix('./dataset/train.txt')

rows = dtrain.num_row()
validation_rate = 0.08
indices = list(range(0, rows))
random.shuffle(indices)
bound_index = int(rows * validation_rate)
validation_index = indices[:bound_index]
train_index = indices[bound_index:]
dvalidate = dtrain.slice(validation_index)
dtrain = dtrain.slice(train_index)

param = {'objective': 'binary:logistic'}

bst = xgb.train(param, dtrain, num_boost_round=10000, evals=[
                (dvalidate, 'eval')], early_stopping_rounds=100)

result = bst.predict(dtest)
pd.Series(result).to_csv(
    path='./output/xgboost.csv',
    header=['label'],
    index_label='id'
)
