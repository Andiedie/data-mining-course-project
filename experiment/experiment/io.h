#pragma once
#include<Eigen/Core>
#include<utility>

const int kFeatureNumber = 201;
const std::string kTrainFilePath = "D:/Andie/code/data-mining-course-project/homework2/dataset/train.txt";
const std::string kTestFilePath = "D:/Andie/code/data-mining-course-project/homework2/dataset/t.txt";
const std::string kTarget = "D:/Andie/code/data-mining-course-project/experiment/output.csv";

std::pair<Eigen::MatrixXd, Eigen::VectorXi> TrainData();
Eigen::MatrixXd TestData();
void SavePrediction(const Eigen::VectorXd & prediction);
