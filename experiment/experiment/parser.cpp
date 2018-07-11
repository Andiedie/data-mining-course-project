#include "parser.h"
#include<iostream>
#include<string>
#include<algorithm>
using std::cerr;
using std::cout;
using std::map;
using std::vector;
using std::string;

map<string, string> ParseArguments(int argc, char* argv[], vector<Argument> arguments) {
	map<string, string> result;
	vector<string> names;
	for (auto &argument : arguments) {
		result[argument.name_] = argument.default_value_;
		names.push_back(argument.name_);
	}
	for (int i = 0; i < argc; i++) {
		string parsed_arg(argv[i]);
		if (parsed_arg == "--help") {
			cout
				<< "Usage: executable [--help]\n";
			for (auto &argument : arguments) {
				cout << "                  [--" << argument.name_ << " " << argument.description_ <<" ]\n";
			}
			exit(0);
		}
		if (parsed_arg.substr(0, 2) == "--") {
			parsed_arg = parsed_arg.substr(2);
			if (std::find(names.begin(), names.end(), parsed_arg) == names.end()) {
				cerr << "Unknown arguments: " << parsed_arg << "\n";
				exit(1);
			} else {
				if (i + 1 < argc) {
					result[parsed_arg] = argv[i + 1];
				} else {
					cerr << "Arguments " << parsed_arg << " requires one argument.\n";
					exit(1);
				}
			}
		}
	}
	return result;
}

Argument::Argument(std::string name, std::string default_value, std::string description)
	: name_(name), default_value_(default_value), description_(description) {}
