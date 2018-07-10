#define _CRT_SECURE_NO_WARNINGS
#include "io.h"
#include<fstream>
#include<string>
#include<streambuf>
#include<omp.h>
#include<vector>
using std::pair;
using std::string;
using std::istreambuf_iterator;
using Eigen::MatrixXd;
using Eigen::VectorXd;

pair<MatrixXd, VectorXd> ReadData(const char *path, bool is_training) {
	std::ifstream file(kTrainFilePath);
	string line;
	std::vector<string> lines;
	while (getline(file, line)) {
		lines.push_back(move(line));
		//if (lines.size() >= 15) break;
	}
	file.close();
	size_t rows = lines.size();
	MatrixXd x = MatrixXd::Zero(rows, kFeatureNumber + 1);
	VectorXd y;
	if (is_training) y.setZero(rows);
	
 #pragma omp parallel for
	for (int row = 0; row < rows; row++) {
		const char *data = lines[row].c_str();
		int remain = int(lines[row].length());
		int label;
		int offset;
		sscanf(data, "%d%n", &label, &offset);
		data += offset;
		if (is_training) y(row) = double(label);
		int index;
		double value;
		x(row, 0) = 1.0;
		while (remain >= 0) {
			sscanf(data, "%d:%lf%n", &index, &value, &offset);
			remain -= offset;
			data += offset;
			x(row, index) = value;
		}
	}
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

