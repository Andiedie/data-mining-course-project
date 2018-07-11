#include<iostream>
#include<chrono>
#include<fstream>
#include"io.h"
#include"logistic-regression.h"
#include"logging.h"
#include"parser.h"
using namespace std;
using namespace Eigen;
using namespace std::chrono;

int main(int argc, char *argv[]) {
	auto arguments = ParseArguments(argc, argv, {
		Argument("train", "train.txt", "train_file_path"),
		Argument("test", "test.txt", "test_file_path"),
		Argument("output", "output.csv", "output_path"),
		Argument("epoch", "100", "train_epochs"),
		Argument("learning_rate", "0.001", "learning_rate"),
		Argument("regularization_parameter", "0", "regularization_parameter"),
	});
	logging::level(logging::Level::kInfo);
	auto beacon = logging::CreateBeacon();

	beacon = logging::CreateBeacon();
	auto train = TrainData(arguments["train"]);
	logging::LogTime(beacon, "loading train data");

	beacon = logging::CreateBeacon();
	auto test_x = TestData(arguments["test"]);
	logging::LogTime(beacon, "loading test data");

	LogisticRegression lr;
	lr.train_epochs(stoi(arguments["epoch"]));
	lr.learning_rate(stod(arguments["learning_rate"]));
	lr.regularization_parameter(stod(arguments["regularization_parameter"]));

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
