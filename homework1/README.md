# Description
This is a project of the Data Mining course (instructor: Dr. Yan Pan) in 2018 for the students in School of Data and Computer Science, Sun Yat-sen University.

This is a linear regression task. Please use what you have learned from our course and write an algorithm to train a regressor.

Have fun!

# Evaluation
The evaluation metric for this competition is RMSE (Root Mean Squared Error).

The file should contain a header and have the following format:

```
id,reference
0,20.1
1,1.232
2,1.3
3,0.855
etc.
```

# Dataset
You are given three files:

## 1.train.csv

The format of each line is as follows:

```
id,value0,value1,...,value383,reference
```
where value0,value1,...,value383 are the features.

## 2.test.csv
This file contains the features whose references you need to predict. The format of each line is as follows:

```
id,value0,value1,...,value383
```

## 3.submission_example.csv
The file you submit should have the same format as this file,the format of each line is as follows:

```
id,reference
```

# Run
## Linear Regression
Use the `tf.estimator.LinearRegressor`.

```bash
python linear_regression.py
```

### Options
- `batch_size`, default `128`
- `train_steps`, default `1000`
- `model_dir`, default `./model/linear_regression`
- `output_path`, default `./linear_regression_output.csv`

## DNN
Use the `tf.estimator.DNNRegressor`.
```bash
python dnn.py
```

### Options
- `batch_size`, default `128`
- `train_steps`, default `1000`
- `model_dir`, default `./model/dnn`
- `output_path`, default `./dnn_output.csv`

## KNN
Use the [k-nearest neighbors algorithm](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm).
```bash
python knn.py
```

### Options
- `batch_size`, default `10`
- `output_path`, default `./knn_output.csv`
- `k`, default `1`

# Performance
| |Linear Regression|DNN|KNN|
|-|-|-|-|
|RMSE|8.48347|0.98436|0.27162|

