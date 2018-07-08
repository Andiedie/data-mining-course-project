#include "tools.h"

pair<MatrixXd, VectorXd> TrainData() {
	// read train data
	// add a column of ones to the feature (bias/intercept term)
	std::ifstream file(kTrainFilePath.c_str());
	int rows = std::count(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(), '\n');
	MatrixXd x(rows, kFeatureNumber + 1);
	VectorXd y(rows);
	file.seekg(0, file.beg);
	string line;
	rows = 0;
	while (getline(file, line)) {
		std::istringstream iss(line);
		string data;
		iss >> data;
		y(rows) = stod(data);
		VectorXd row = VectorXd::Zero(kFeatureNumber + 1);
		while (iss >> data) {
			size_t pos = data.find(':');
			int index = stoi(data.substr(0, pos));
			double value = stod(data.substr(pos + 1));
			row(index) = value;
		}
		row(0) = 1.0;
		x.row(rows) = row;
		rows++;
	}
	return pair<MatrixXd, VectorXd>(move(x), move(y));
}

MatrixXd TestData() {
	// read train data
	// add a column of ones to the feature (bias/intercept term)
	std::ifstream file(kTestFilePath.c_str());
	int rows = std::count(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(), '\n');
	MatrixXd x(rows, kFeatureNumber + 1);
	file.seekg(0, file.beg);
	string line;
	rows = 0;
	while (getline(file, line)) {
		std::istringstream iss(line);
		string data;
		iss >> data;
		VectorXd row = VectorXd::Zero(kFeatureNumber + 1);
		while (iss >> data) {
			size_t pos = data.find(':');
			int index = stoi(data.substr(0, pos));
			double value = stod(data.substr(pos + 1));
			row(index) = value;
		}
		x.row(rows) = row;
		rows++;
	}
	return x;
}
