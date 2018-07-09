#include"logistic-regression.h"
#include<iostream>
using namespace std;

LogisticRegression::LogisticRegression() {
	learning_rate_ = 0.001;
	train_epochs_ = 100;
	regularization_parameter_ = 0;
}

LogisticRegression::LogisticRegression(double learning_rate, double regularization_parameter, int train_epochs)
	: learning_rate_(learning_rate), regularization_parameter_(regularization_parameter), train_epochs_(train_epochs) {}

double LogisticRegression::learning_rate() const {
	return learning_rate_;
}

void LogisticRegression::learning_rate(double learning_rate) {
	this->learning_rate_ = learning_rate;
}

int LogisticRegression::train_epochs() const {
	return train_epochs_;
}

void LogisticRegression::train_epochs(int train_epochs) {
	this->train_epochs_ = train_epochs;
}

double LogisticRegression::regularization_parameter() const {
	return regularization_parameter_;
}

void LogisticRegression::regularization_parameter(double regularization_parameter) {
	this->regularization_parameter_ = regularization_parameter;
}

void LogisticRegression::SerialTrain(const MatrixXd &x, const VectorXd &y) {
	size_t num_examples = x.rows();
	size_t num_features = x.cols();
	theta_.setZero(num_features);

	for (size_t epoch = 0; epoch < train_epochs_; epoch++) {
		cout << "epoch " << epoch << endl;
		VectorXd gradient = VectorXd::Zero(num_features);
		for (size_t i = 0; i < num_examples; i++) {
			auto gradient_i = (Sigmoid(x.row(i) * theta_) - y(i)) * x.row(i);
			gradient += gradient_i;
		}
		regularize(gradient);
		gradient *= 1.0 / num_examples;
		theta_ -= learning_rate_ * gradient;
	}
}

void LogisticRegression::ParallelTrain(const MatrixXd &x, const VectorXd &y) {
}

VectorXd LogisticRegression::PredictProbability(MatrixXd &x) const {
	VectorXd result = (x * theta_).unaryExpr(std::ptr_fun(LogisticRegression::Sigmoid));
	return move(result);
}

double LogisticRegression::Sigmoid(double x) {
	return 1.0 / (1.0 + std::exp(-x));
}

void LogisticRegression::regularize(VectorXd & gradient) {
	size_t length = gradient.rows();
	VectorXd theta_without_bias = theta_.block(1, 0, length - 1, 1);
	gradient.block(1, 0, length - 1, 1) += regularization_parameter_ * theta_without_bias;
}
