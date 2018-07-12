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
		Argument("train", "D:/Andie/code/data-mining-course-project/homework2/dataset/train.txt", "train_file_path"),
		Argument("test", "D:/Andie/code/data-mining-course-project/homework2/dataset/test.txt", "test_file_path"),
		Argument("output", "D:/Download/output.csv", "output_path"),
		Argument("epoch", "100", "train_epochs"),
		Argument("learning_rate", "0.001", "learning_rate"),
		Argument("regularization_parameter", "0", "regularization_parameter"),
		Argument("method", "p", "how_to_train"),
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

	beacon = logging::CreateBeacon();
	switch (arguments["method"][0]) {
	case 'p':
		lr.ParallelTrain(train.first, train.second);
		logging::LogTime(beacon, "parallel training");
		break;
	case 's':
		lr.SerialTrain(train.first, train.second);
		logging::LogTime(beacon, "serial training");
		break;
	case 'u':
		lr.CacheUnfriendlyParallelTrain(train.first, train.second);
		logging::LogTime(beacon, "cache unfriendly parallel training");
		break;
	default:
		logging::Error() << "Unknown training method, only 'p', 's', 'u' is acceptable";
		break;
	}
	
	beacon = logging::CreateBeacon();
	auto result = lr.PredictProbability(test_x);
	logging::LogTime(beacon, "prediction");

	SavePrediction(result, arguments["output"]);
	return 0;
}
