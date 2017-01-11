set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations -Wreturn-type -Wuninitialized -Wunused-variable -std=c++0x")

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")

################  THIS IS THE SIGNALTAP LIBRARY
add_library(kalman-filter SHARED
  kalman_filter.cpp
  KF_Models.cpp)

############## Define test executables
#add_executable(kf_unittest test_kalman_filter.cpp)




