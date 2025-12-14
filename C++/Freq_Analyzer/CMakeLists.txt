cmake_minimum_required(VERSION 4.0)
project(Freq_Analyzer)

set(CMAKE_CXX_STANDARD 23)

add_executable(Freq_Analyzer main.cpp
        SignalGenerator.cpp
        SignalGenerator.h
        Analyzer.cpp
        Analyzer.h
        hw/Timer.cpp
        hw/Timer.h
        hw/ADCBuffer.cpp
        hw/ADCBuffer.h
        cmake-build-debug/PWMController.cpp
        cmake-build-debug/PWMController.h)
