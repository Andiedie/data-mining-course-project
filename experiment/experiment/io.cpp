#include "io.h"
#include<fstream>
#include<sstream>
#include<string>
using std::pair;
using std::string;
using Eigen::MatrixXd;
using Eigen::VectorXd;

pair<MatrixXd, VectorXd> ReadData(const char *path, bool is_training) {
	std::ifstream file(kTrainFilePath);
	size_t rows = std::count(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(), '\n');
	MatrixXd x = MatrixXd::Zero(rows, kFeatureNumber + 1);
	VectorXd y;
	if (is_training) y.setZero(rows);
	file.seekg(0, file.beg);
	string line;
	rows = 0;
	while (getline(file, line)) {
		std::istringstream iss(line);
		string data;
		iss >> data;
		if (is_training) y(rows) = stod(data);
		x(rows, 0) = 1.0;
		while (iss >> data) {
			size_t pos = data.find(':');
			int index = stoi(data.substr(0, pos));
			double value = stod(data.substr(pos + 1));
			x(rows, index) = value;
		}
		rows++;
	}
	file.close();
	return pair<MatrixXd, VectorXd>(std::move(x), std::move(y));
}

pair<MatrixXd, VectorXd> TrainData() {
	return ReadData(kTrainFilePath.c_str(), true);
}

MatrixXd TestData() {
	return ReadData(kTestFilePath.c_str(), false).first;
}

void SavePrediction(const VectorXd & prediction) {
	std::ofstream target(kTarget.c_str());
	target << "id,label\n";
	size_t rows = prediction.rows();
	for (int i = 0; i < rows; i++) {
		target << i << "," << prediction(i) << '\n';
	}
	target.close();
}

