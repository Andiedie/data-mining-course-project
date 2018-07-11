#include<iostream>
#include<chrono>
#include<fstream>
#include"io.h"
#include"logistic-regression.h"
#include"logging.h"
using namespace std;
using namespace Eigen;
using namespace std::chrono;

int main(int argc, char *argv[]) {
	auto arguments = ParseArguments(argc, argv);
	logging::level(logging::Level::kInfo);
	auto beacon = logging::CreateBeacon();

	beacon = logging::CreateBeacon();
	auto train = TrainData(arguments["train"]);
	logging::LogTime(beacon, "loading train data");

	beacon = logging::CreateBeacon();
	auto test_x = TestData(arguments["test"]);
	logging::LogTime(beacon, "loading test data");

	LogisticRegression lr;
	lr.train_epochs(10);
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

	SavePrediction(result, arguments["output"]);
	return 0;
}
