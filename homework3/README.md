# Description
This is a project of the Data Mining course (instructor: Dr. Yan Pan) in 2018 for the students in School of Data and Computer Science, Sun Yat-sen University.

This is a image classification task. Please use what you have learned from our course and write an algorithm to train a classifier.

There are 21048 training examples and 4500 testing examples of 8 classes.

Have fun!

# Evaluation
The evaluation metric for this competition is Categorization Accuracy, which is equal to accuracy of your predictions (the percentage of sequences where you predict the next number correctly).

## Submission Format
The file you submit should have the same format as this file,the format of each line is as follows:

```
Image,Cloth_label
img4.jpg,2
img5.jpg,1
img6.jpg,3
...
```

where Image is the file name of a test image which can be found in 'image/test' folder, and cloth_label is the result.

# Dataset
You are given two files:

## 1.image.rar
contains train images and test images

`image/train`: the folder contains images of train

`image/test`: the folder contains images of test

## 2.train.csv
The train.csv file contains the image and cloth_label. The format of each line is as follows:

```
Image,Cloth_label
img1.jpg,0
img2.jpg,1
img3.jpg,7
...
```

where Image is the file name of a train image, which can be found in `image/train` folder.

# Run
## Resnet
Use restnet50 by default.

```bash
python main.py
```

### Options
- `predict`, to predict, you should give the output file name.

# Performance
| |Resnet50|
|-|-|
|AUC|0.78370|
