set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
SET(CMAKE_EXE_LINKER_FLAGS  "-qmkl ${CMAKE_EXE_LINKER_FLAGS}")
include_directories(${CMAKE_SOURCE_DIR}/03_sycl_migrated/Common/)

add_executable (03_sycl_migrated src/oceanFFT.cpp src/oceanFFT_kernel.cpp)
target_link_libraries(03_sycl_migrated sycl)

add_custom_target (run_cpu cd ${CMAKE_SOURCE_DIR}/03_sycl_migrated/ && ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/03_sycl_migrated)
add_custom_target (run_gpu SYCL_DEVICE_FILTER=gpu cd ${CMAKE_SOURCE_DIR}/03_sycl_migrated/ && ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/03_sycl_migrated)