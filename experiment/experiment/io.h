#pragma once
#include<fstream>
#include<sstream>
#include<utility>
#include<string>
#include<Eigen/Core>
using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::string;
using std::pair;
using std::move;

const int kFeatureNumber = 201;
const string kTrainFilePath = "D:/Andie/code/data-mining-course-project/homework2/dataset/sample_train.txt";
const string kTestFilePath = "D:/Andie/code/data-mining-course-project/homework2/dataset/sample_test.txt";
const string kTarget = "D:/Andie/code/data-mining-course-project/experiment/output.csv";

pair<MatrixXd, VectorXd> TrainData();
MatrixXd TestData();
void SavePrediction(const VectorXd & prediction);
