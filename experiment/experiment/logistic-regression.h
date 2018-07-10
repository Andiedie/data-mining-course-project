#pragma once
#include<Eigen/Core>

class LogisticRegression {
private:
	double learning_rate_;
	int train_epochs_;
	double regularization_parameter_;
	Eigen::VectorXd theta_;
public:
	LogisticRegression();
	LogisticRegression(double learning_rate, double regularization_parameter, int train_epoch);

	double learning_rate() const;
	void learning_rate(double learning_rate);
	int train_epochs() const;
	void train_epochs(int train_epochs);
	double regularization_parameter() const;
	void regularization_parameter(double regularization_parameter);

	void SerialTrain(const Eigen::MatrixXd & x, const Eigen::VectorXd & y);
	void ParallelTrain(const Eigen::MatrixXd & x, const Eigen::VectorXd & y);

	Eigen::VectorXd PredictProbability(Eigen::MatrixXd & x) const;
private:
	static double Sigmoid(double x);
	void regularize(Eigen::VectorXd & gradient);
};
