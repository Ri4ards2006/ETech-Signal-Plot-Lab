cmake_minimum_required(VERSION 4.0)
project(C__)

set(CMAKE_CXX_STANDARD 23)

add_executable(C__ src/main.cpp
        src/SignalBuffer.cpp
        include/SignalBuffer.h
        src/Analyzer.cpp
        include/Analyzer.h)
