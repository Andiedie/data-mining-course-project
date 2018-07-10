#include<iostream>
#include"io.h"
#include"logistic-regression.h"
#include"logger.h"
using namespace std;

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

int main() {
	//auto train = TrainData();
	auto train = load_csv("D:/Download/train.csv", 559, 30);
	LogisticRegression lr;
	lr.train_epochs(200);
	lr.learning_rate(0.001);
	lr.regularization_parameter(1);
	lr.SerialTrain(train.first, train.second);
	auto test = load_csv("D:/Download/test.csv", 10, 30);
	//auto test_x = TestData();
	auto result = lr.PredictProbability(test.first);
	cout << "predict: " << result.transpose() << '\n';
	cout << " actual: " << test.second.transpose() << '\n';
	// accuracy
	Eigen::ArrayXd accuracy = Eigen::ArrayXd(result - test.second).abs();
	cout << "accuracy: " << (accuracy < 0.5).count() / (accuracy.size() + 0.0) << "\n";
	// save result
	SavePrediction(result);
	return 0;
}
