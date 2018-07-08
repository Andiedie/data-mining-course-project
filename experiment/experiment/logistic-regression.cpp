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

void LogisticRegression::SerialTrain(const MatrixXd &x, const VectorXd &y) {
	int num_examples = x.rows();
	int num_features = x.cols();
	theta_.setZero();

	for (size_t i = 0; i < train_epochs_; i++) {
		VectorXd z = theta_ * x;
		VectorXd hypothesis = Sigmoid(z);
		VectorXd gradient = (1.0 / num_examples) * ()
	}
}

void LogisticRegression::ParallelTrain(const MatrixXd &x, const VectorXd &y) {
}

VectorXd LogisticRegression::PredictProbability(VectorXd &x) const {
	return VectorXd();
}

int LogisticRegression::Predict(VectorXd &x) const {
	return 0;
}

VectorXd LogisticRegression::Sigmoid(VectorXd & v) const {
	return 1.0 / ((-v).exp() + 1.0);
}
