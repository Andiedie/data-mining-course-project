#include "logger.h"
using logging::Level;

const char* LevelName(Level level) {
	switch (level) {
	case Level::kDebug:
		return "DEBUG";
	case Level::kInfo:
		return "INFO";
	case Level::kWarn:
		return "WARN";
	case Level::kError:
		return "ERROR";
	default:
		return "";
	}
}

class NullBuffer : public std::streambuf {
public:
	int overflow(int c) { return c; }
};

NullBuffer null_buffer;
std::ostream null_stream(&null_buffer);
Level level_ = Level::kDebug;
std::ostream &out_ = std::cout;

Level logging::level() {
	return level_;
}

void logging::level(Level level) {
	level_ = level;
}

std::ostream & logging::debug() {
	return log(Level::kDebug);
}

std::ostream & logging::info() {
	return log(Level::kInfo);
}

std::ostream & logging::warn() {
	return log(Level::kWarn);
}

std::ostream & logging::error() {
	return log(Level::kError);
}

std::ostream & logging::log(Level level) {
	if (level >= level_) {
		auto raw_time = std::time(nullptr);
		std::tm time_info{};
#if defined(__unix__)
		localtime_r(&raw_time, &time_info);
#elif defined(_MSC_VER)
		localtime_s(&time_info, &raw_time);
#else
		static std::mutex mtx;
		std::lock_guard<std::mutex> lock(mtx);
		time_info = *std::localtime(&raw_time);
#endif
		auto time = std::put_time(&time_info, "%Y-%m-%d %H:%M:%S");
		out_ << time << " [" << LevelName(level) << "] ";
		return out_;
	}
	return null_stream;
}
