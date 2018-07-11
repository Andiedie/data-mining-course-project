#pragma once
#include<map>
#include<vector>

class Argument {
public:
	Argument(std::string name, std::string default_value, std::string description);
	std::string name_;
	std::string default_value_;
	std::string description_;
};

std::map<std::string, std::string> ParseArguments(int argc, char* argv[], std::vector<Argument> arguments);
