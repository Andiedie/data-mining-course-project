#include<iostream>
#include<chrono>
#include"io.h"
#include"logistic-regression.h"
#include"logging.h"
using namespace std;
using namespace Eigen;
using namespace std::chrono;

int main() {
	logging::level(logging::Level::kDebug);
	auto train = TrainData();
	logging::Info() << "train data loaded\n";
	//auto train = load_csv("D:/Download/train.csv", 559, 30);
	LogisticRegression lr;
	lr.train_epochs(5);
	lr.learning_rate(0.001);
	lr.regularization_parameter(1);
	auto beacon = logging::CreateBeacon();
	lr.ParallelTrain(train.first, train.second);
	logging::LogTime(beacon, "training");
	//auto test = load_csv("D:/Download/test.csv", 10, 30);
	//auto test_x = TestData();
	//logging::info() << "test data loaded\n";
	//auto result = lr.PredictProbability(test_x);
	//logging::info() << "predict: " << result.transpose() << '\n';
	//logging::info() << "actual: " << test_x.transpose() << '\n';
	// accuracy
	//Eigen::ArrayXd accuracy = Eigen::ArrayXd(result - test_x).abs();
	//logging::info() << "accuracy: " << (accuracy < 0.5).count() / (accuracy.size() + 0.0) << "\n";
	// save result
	//SavePrediction(result);
	return 0;
}
