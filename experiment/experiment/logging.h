#pragma once
#include<ostream>

namespace logging {
	enum Level {
		kDebug,
		kInfo,
		kWarn,
		kError
	};
	Level level();
	void level(Level level);
	std::ostream &debug();
	std::ostream &info();
	std::ostream &warn();
	std::ostream &error();
	std::ostream &log(Level level);
}
