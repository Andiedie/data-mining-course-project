# Description
This is a project of the Data Mining course (instructor: Dr. Yan Pan) in 2018 for the students in School of Data and Computer Science, Sun Yat-sen University.

This is a classification task. Please use what you have learned from our course and write an algorithm to train a classifier.

There are 2 classes and 201-dimensional features for each sample. There are 1719692 training examples and 429923 testing examples.

Have fun!

# Evaluation
The evaluation metric for this competition is AUC(Area under Receiver Operating Characteristic Curve)[binary-classification].

The file should contain a header and have the following format:

```
id,reference
id,label
0,0.01
1,0.99
2,0.32
etc.
```

# Dataset
You are given three files:

## 1.train.csv
The format of each line is as follows:

```
label index1:value1 index3:value3 index4:value4 ...
```
where index1,index3,...is the features and value1,value3,... are value of the features(the ignored value equals 0 e.g. the value2 here equals 0) and there are only 2 classes(label) in all,indexed from 0 to 1.

## 2.test.csv
This file contains the features without labels that you need to predict. And the format of data is the same as the train.txt. The format of each line is as follows:

```
id index1:value1 index2:value2 index3:value3 ...
```

## 3.sample_submission.txt
The file you submit should have the same format as this file,the format of each line is as follows:

```
id,label
```

# Run
## sklearn
Use the 9 classification algorithm of `scikit-learn`.

```bash
python sklearn_.py
```

### Options
- `algorithm`, use the specific algorithm, default `None` means using all.

## xgboost
```
python xgboost_.py
```

# Performance
| |Nearest Neighbors|Decision Tree|Random Forest|Neural Net|
|-|-|-|-|-|
|AUC|0.80139|0.88573|0.39052|0.87256|

| |AdaBoost|Naive Bayes|QDA|xgboost|
|-|-|-|-|-|
|AUC|0.89531|0.73119|0.49650|0.90236|
