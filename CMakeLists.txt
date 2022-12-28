cmake_minimum_required(VERSION 3.14)
project(fast_pyrdown C CXX)
set(CMAKE_CXX_STANDARD 14)

LIST(APPEND CMAKE_CXX_FLAGS
        "${CMAKE_CXX_FLAGS}
        -std=c++0x
        -mfpu=neon -mfloat-abi=hard")

#add_definitions("-DCPLUS_INCLUDE_PATH=/usr/local/opt/llvm/include/c++/v1:/Library/Developer/CommandLineTools/SDKs/MacOSX13.sdk/usr/include/")
#set(CPLUS_INCLUDE_PATH=/usr/local/opt/llvm/include/c++/v1:/Library/Developer/CommandLineTools/SDKs/MacOSX13.sdk/usr/include/)
#include_directories("/usr/local/opt/llvm/include/c++/v1")
#include_directories("/Library/Developer/CommandLineTools/SDKs/MacOSX13.sdk/usr/include/")
#include_directories("/Applications/Xcode_14_2.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include/c++/v1")
find_package(OpenCV REQUIRED core imgproc imgcodecs highgui)
find_package(Eigen3 REQUIRED)

if (POLICY CMP0042)
    cmake_policy (SET CMP0042 NEW)
endif ()

add_definitions(-DEIGEN_TEST_NEON=ON)
#set(SRC fast_pyrdown)
#add_subdirectory(eigen)
aux_source_directory("fast_pyrdown" SRC)
add_executable(fast_pyrdown ${SRC})

target_include_directories(fast_pyrdown PUBLIC "${EIGEN3_INCLUDE_DIR}")
target_link_libraries(fast_pyrdown ${OpenCV_LIBS})
