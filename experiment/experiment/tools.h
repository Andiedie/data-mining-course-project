#pragma once
#include<fstream>
#include<sstream>
#include<utility>
#include<string>
#include<Eigen/Dense>
using namespace std;
using namespace Eigen;

const int kFeatureNumber = 201;
const string kTrainFilePath = "D:/Andie/code/data-mining-course-project/homework2/dataset/t.txt";
const string kTestFilePath = "D:/Andie/code/data-mining-course-project/homework2/dataset/t.txt";

pair<MatrixXd, VectorXd> TrainData();
MatrixXd TestData();
