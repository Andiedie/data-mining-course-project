#pragma once
#include<ostream>
#include<chrono>

namespace logging {
	enum Level {
		kDebug,
		kInfo,
		kWarn,
		kError
	};
	Level level();
	void level(Level level);
	std::ostream &Debug();
	std::ostream &Info();
	std::ostream &Warn();
	std::ostream &Error();
	std::ostream &Log(Level level);
	std::chrono::high_resolution_clock::time_point CreateBeacon();
	void LogTime(std::chrono::high_resolution_clock::time_point beacon, const char* name, Level level = Level::kInfo);
}
