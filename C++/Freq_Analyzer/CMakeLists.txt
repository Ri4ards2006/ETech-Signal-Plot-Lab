cmake_minimum_required(VERSION 4.0)
project(Freq_Analyzer)

set(CMAKE_CXX_STANDARD 23)

add_executable(Freq_Analyzer src/main.cpp
        src/signal/SignalGenerator.cpp
        src/signal/SignalGenerator.h
        src/dsp/Analyzer.cpp
        src/dsp/Analyzer.h
        src/hw/Timer.cpp
        src/hw/Timer.h
        src/hw/ADCBuffer.cpp
        src/hw/ADCBuffer.h
        cmake-build-debug/PWMController.cpp
        cmake-build-debug/PWMController.h
        src/dsp/FFT.cpp
        src/dsp/FFT.h
        src/dsp/Window.cpp
        src/dsp/Window.h
        src/buffer/RingBuffer.cpp
        src/buffer/RingBuffer.h)
