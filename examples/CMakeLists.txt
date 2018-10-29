cmake_minimum_required(VERSION 3.1)
set (CMAKE_CXX_STANDARD 11)
project(myopencv_examples)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(example01_canny example01_canny.cpp)
target_link_libraries(example01_canny ${OpenCV_LIBS})

add_executable(example02_slider example02_slider.cpp)
target_link_libraries(example02_slider ${OpenCV_LIBS})

add_executable(example03_video example03_video.cpp)
target_link_libraries(example03_video ${OpenCV_LIBS})

add_executable(example04_sharpen example04_sharpen.cpp)
target_link_libraries(example04_sharpen ${OpenCV_LIBS})

#add_executable(example05_video_homography example05_video_homography.cpp)
#target_link_libraries(example05_video_homography ${OpenCV_LIBS})

add_executable(example06_retina example06_retina.cpp)
target_link_libraries(example06_retina ${OpenCV_LIBS})

add_executable(example07_dnn_objdetect example07_dnn_objdetect.cpp)
target_link_libraries(example07_dnn_objdetect ${OpenCV_LIBS})
