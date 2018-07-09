#include<iostream>
#include"io.h"
#include"logistic-regression.h"
using namespace std;

int main() {
	auto train = TrainData();
	LogisticRegression lr;
	lr.train_epochs(5);
	lr.SerialTrain(train.first, train.second);
	auto test_x = TestData();
	cout << lr.PredictProbability(test_x);
	system("pause");
}
