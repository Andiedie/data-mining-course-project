#include "logging.h"
#include<mutex>
#include<iostream>
#include<ctime>
#include<iomanip>
using logging::Level;
using namespace std::chrono;

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

std::ostream & logging::Debug() {
	return Log(Level::kDebug);
}

std::ostream & logging::Info() {
	return Log(Level::kInfo);
}

std::ostream & logging::Warn() {
	return Log(Level::kWarn);
}

std::ostream & logging::Error() {
	return Log(Level::kError);
}

std::ostream & logging::Log(Level level) {
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
		out_ << "\n" << time << " [" << LevelName(level) << "] ";
		return out_;
	}
	return null_stream;
}

high_resolution_clock::time_point logging::CreateBeacon() {
	return high_resolution_clock::now();
}

void logging::LogTime(high_resolution_clock::time_point beacon, const char* name, Level level) {
	auto now = high_resolution_clock::now();
	auto duration = duration_cast<milliseconds>(now - beacon).count();
	Log(level) << name << " takes " << duration << " ms";
}
