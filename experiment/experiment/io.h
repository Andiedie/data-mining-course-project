#pragma once
#include<Eigen/Core>
#include<utility>
#include<map>

const int kFeatureNumber = 201;
const std::string kTrainFilePath = "train.txt";
const std::string kTestFilePath = "test.txt";
const std::string kTarget = "output.csv";

std::pair<Eigen::MatrixXd, Eigen::VectorXi> TrainData(std::string path);
Eigen::MatrixXd TestData(std::string path);
void SavePrediction(const Eigen::VectorXd & prediction, std::string path);
std::map<std::string, std::string> ParseArguments(int argc, char* argv[]);
