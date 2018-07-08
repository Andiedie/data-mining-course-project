#pragma once
#include<Eigen/Dense>
using Eigen::VectorXd;
using Eigen::MatrixXd;

class LogisticRegression {
private:
	double learning_rate_;
	int train_epochs_;
	VectorXd theta_;
public:
	LogisticRegression();
	double learning_rate() const;
	void set_learning_rate(double learning_rate);
	int train_epochs() const;
	void set_train_epochs(int train_epochs);
	void SerialTrain(const MatrixXd & x, const VectorXd & y);
	void ParallelTrain(const MatrixXd & x, const VectorXd & y);
	VectorXd PredictProbability(VectorXd & x) const;
	int Predict(VectorXd & x) const;
private:
	VectorXd Sigmoid(VectorXd & v) const;
};
