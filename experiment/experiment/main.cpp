#include<iostream>
#include<chrono>
#include<fstream>
#include"io.h"
#include"logistic-regression.h"
#include"logging.h"
using namespace std;
using namespace Eigen;
using namespace std::chrono;

std::pair<MatrixXd, VectorXd> load_csv(const char *filename, const int row, const int col);

int main() {
	logging::level(logging::Level::kInfo);
	auto train = TrainData();
	//auto train = load_csv("D:/Download/train.csv", 559, 30);
	logging::Info() << "train data loaded";
	LogisticRegression lr;
	lr.train_epochs(20);
	lr.learning_rate(0.001);
	lr.regularization_parameter(1);
	auto beacon = logging::CreateBeacon();
	lr.ParallelTrain(train.first, train.second);
	logging::LogTime(beacon, "training");
	//auto test = load_csv("D:/Download/test.csv", 10, 30);
	//auto &test_x = test.first;
	//auto test_x = TestData();
	//logging::Info() << "test data loaded";
	//auto result = lr.PredictProbability(test_x);
	//logging::Info() << "predict: " << result.transpose();
	//logging::Info() << "actual: " << test.second.transpose();
	// // accuracy
	//Eigen::ArrayXd accuracy = Eigen::ArrayXd(result - test.second).abs();
	//logging::Info() << "accuracy: " << (accuracy < 0.5).count() / (accuracy.size() + 0.0);
	// save result
	//SavePrediction(result);
	return 0;
}

std::pair<MatrixXd, VectorXd> load_csv(const char *filename, const int row, const int col) {
	MatrixXd data(row, col + 1);
	VectorXd tags(row);

	ifstream f(filename);
	assert(f.good());

	string line, tmp;
	for (size_t i = 0; i < row; i++) {
		std::getline(f, line);
		stringstream ss(line);

		data(i, 0) = 1;
		for (size_t j = 1; j <= col; j++) {
			std::getline(ss, tmp, ',');
			data(i, j) = std::stod(tmp);
		}

		std::getline(ss, tmp, ',');
		tags(i) = std::stoi(tmp);
	}

	return std::make_pair(data, tags);
}
