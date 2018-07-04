#include"logistic-regression.h"

LogisticRegression::LogisticRegression() {
	learning_rate_ = 0.001;
	train_epochs_ = 100;
}

double LogisticRegression::learning_rate() const {
	return learning_rate_;
}

void LogisticRegression::set_learning_rate(double learning_rate) {
	this->learning_rate_ = learning_rate;
}

int LogisticRegression::train_epochs() const {
	return train_epochs_;
}

void LogisticRegression::set_train_epochs(int train_epochs) {
	this->train_epochs_ = train_epochs;
}

void LogisticRegression::SerialTrain(MatrixXd x, VectorXd y) {
	// add a column of ones to x as bias
	

}

void LogisticRegression::ParallelTrain(MatrixXd x, VectorXd y) {
}

VectorXd LogisticRegression::PredictProbability(VectorXd x) const {
	return VectorXd();
}

int LogisticRegression::Predict(VectorXd x) const {
	return 0;
}
