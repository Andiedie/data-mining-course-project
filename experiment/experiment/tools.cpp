#include "tools.h"
#include<iostream>

pair<MatrixXd, VectorXd> TrainData() {
	ifstream file(kTrainFilePath.c_str());
	int rows = count(istreambuf_iterator<char>(file), istreambuf_iterator<char>(), '\n');
	MatrixXd x(rows, kFeatureNumber);
	VectorXd y(rows);
	file.seekg(0, file.beg);
	string line;
	rows = 0;
	while (getline(file, line)) {
		istringstream iss(line);
		string data;
		iss >> data;
		y(rows) = stod(data);
		VectorXd row = VectorXd::Zero(kFeatureNumber);
		while (iss >> data) {
			size_t pos = data.find(':');
			int index = stoi(data.substr(0, pos)) - 1;
			double value = stod(data.substr(pos + 1));
			row(index) = value;
		}
		x.row(rows) = row;
		rows++;
	}
	return pair<MatrixXd, VectorXd>(move(x), move(y));
}

MatrixXd TestData() {
	ifstream file(kTestFilePath.c_str());
	int rows = count(istreambuf_iterator<char>(file), istreambuf_iterator<char>(), '\n');
	cout << rows << endl;
	MatrixXd x(rows, kFeatureNumber);
	file.seekg(0, file.beg);
	string line;
	rows = 0;
	while (getline(file, line)) {
		istringstream iss(line);
		string data;
		iss >> data;
		VectorXd row = VectorXd::Zero(kFeatureNumber);
		while (iss >> data) {
			size_t pos = data.find(':');
			int index = stoi(data.substr(0, pos)) - 1;
			double value = stod(data.substr(pos + 1));
			row(index) = value;
		}
		x.row(rows) = row;
		rows++;
	}
	return x;
}
