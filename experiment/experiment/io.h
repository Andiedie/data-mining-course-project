#pragma once
#include<Eigen/Core>
#include<utility>

const int kFeatureNumber = 201;

std::pair<Eigen::MatrixXd, Eigen::VectorXi> TrainData(std::string path);
Eigen::MatrixXd TestData(std::string path);
void SavePrediction(const Eigen::VectorXd & prediction, std::string path);
