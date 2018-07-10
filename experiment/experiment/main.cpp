#include<iostream>
#include<chrono>
#include<fstream>
#include"io.h"
#include"logistic-regression.h"
#include"logging.h"
using namespace std;
using namespace Eigen;
using namespace std::chrono;

int main() {
	logging::level(logging::Level::kDebug);
	auto beacon = logging::CreateBeacon();

	beacon = logging::CreateBeacon();
	auto train = TrainData();
	logging::LogTime(beacon, "loading train data");

	beacon = logging::CreateBeacon();
	auto test_x = TestData();
	logging::LogTime(beacon, "loading test data");

	LogisticRegression lr;
	lr.train_epochs(100);
	lr.learning_rate(0.001);
	lr.regularization_parameter(1);

	//auto beacon = logging::CreateBeacon();
	//lr.SerialTrain(train.first, train.second);
	//logging::LogTime(beacon, "serial training");

	beacon = logging::CreateBeacon();
	lr.ParallelTrain(train.first, train.second);
	logging::LogTime(beacon, "parallel training");
	
	beacon = logging::CreateBeacon();
	auto result = lr.PredictProbability(test_x);
	logging::LogTime(beacon, "prediction");

	//Eigen::ArrayXd accuracy = Eigen::ArrayXd(result - test.second).abs();
	//logging::Info() << "accuracy: " << (accuracy < 0.5).count() / (accuracy.size() + 0.0);

	SavePrediction(result);
	return 0;
}
