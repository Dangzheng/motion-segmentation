#Project
cmake_minimum_required(VERSION 2.4)
project(test)

#Flags
# set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -Wall -g -ggdb -msse4.2 -stdlib=libc++")
# set(CMAKE_BUILD_TYPE Debug)
if (APPLE)
  #   set(CMAKE_CXX_COMPILER "/usr/bin/clang++")
  #   set(CMAKE_C_COMPILER "/usr/bin/clang")
   set(CMAKE_CXX_COMPILER clang-omp++ CACHE STRING "C++ compiler" FORCE)
   set(CMAKE_C_COMPILER clang-omp CACHE STRING "C compiler" FORCE)
   set(CMAKE_CXX_FLAGS_RELEASE "-Wall -O3 -msse4.2 -stdlib=libstdc++ -fopenmp")
   set(CMAKE_C_FLAGS_RELEASE "-Wall -O3 -msse4.2 -stdlib=libstdc++ -fopenmp")
   add_compile_options(-std=c++11)
endif (APPLE)
# Flow
# find_package(OpenMP REQUIRED)
FIND_PACKAGE(OpenMP)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
find_package(OpenCV REQUIRED)
file(GLOB TEST_SRC_FILES "*.cpp")
add_executable(test ${TEST_SRC_FILES})
include_directories(/usr/local/Cellar/eigen/3.2.9/include/eigen3)
include_directories(/usr/local/include)
target_link_libraries(test ${OpenCV_LIBS} png)

#usage
#CC=clang-omp CXX=clang-omp++ cmake .

