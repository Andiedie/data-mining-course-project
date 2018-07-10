#pragma once
#include<Eigen/Core>
#include<thread>
using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::thread;

class LogisticRegression {
private:
	double learning_rate_;
	int train_epochs_;
	double regularization_parameter_;
	VectorXd theta_;
public:
	LogisticRegression();
	LogisticRegression(double learning_rate, double regularization_parameter, int train_epoch);

	double learning_rate() const;
	void learning_rate(double learning_rate);
	int train_epochs() const;
	void train_epochs(int train_epochs);
	double regularization_parameter() const;
	void regularization_parameter(double regularization_parameter);

	void SerialTrain(const MatrixXd & x, const VectorXd & y);
	void ParallelTrain(const MatrixXd & x, const VectorXd & y);

	VectorXd PredictProbability(MatrixXd & x) const;
private:
	static double Sigmoid(double x);
	void regularize(VectorXd & gradient);
};
