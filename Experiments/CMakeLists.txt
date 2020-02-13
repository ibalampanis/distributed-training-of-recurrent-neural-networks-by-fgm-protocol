cmake_minimum_required(VERSION 2.6)
project(Distributed_Training_of_Recurrent_Neural_Networks_by_FGM_protocol)

option(DEBUG "DEBUG" OFF)

set(MODEL_SOURCE_DIR .)
include_directories("..")

set(SOURCES ${MODEL_SOURCE_DIR}/ ../RNN_models/predictor/RNNPredictor.cc ../RNN_models/predictor/RNNPredictor.hh CentrEEG.cc)

if (DEBUG)
    message("Compilation with debug info (with ggdb3 flag)")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb3")
else ()
    message("Compilation without debug info (without ggdb3 flag)")
endif ()


add_executable(CentrEEG ${SOURCES})
target_link_libraries(CentrEEG
        ${MLPACK_LIBRARIES}
        ${ARMADILLO_LIBRARIES}
        ${Boost_LIBRARIES})
#Enable OpenMP package
find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif ()

if (NOT EXISTS ${PROJECT_BINARY_DIR}/../data)
    add_custom_command(TARGET CentrEEG POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory
            ${PROJECT_BINARY_DIR}/../data)
endif ()


if (NOT EXISTS ${PROJECT_BINARY_DIR}/../../../saved_models)
    add_custom_command(TARGET CentrEEG POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory
            ${PROJECT_BINARY_DIR}/../../saved_models)
endif ()


if (NOT EXISTS ${PROJECT_BINARY_DIR}/../data/EEG_Eye_State.csv)
    add_custom_command(TARGET CentrEEG POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_SOURCE_DIR}/../../data/EEG_Eye_State.csv
            ${PROJECT_BINARY_DIR}/../data/EEG_Eye_State.csv)
endif ()

