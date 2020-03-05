cmake_minimum_required(VERSION 2.6)
project(Distributed_Training_of_Recurrent_Neural_Networks_by_FGM_protocol)
set(DCMAKE_CXX_COMPILER "g++-7")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

option(DEBUG "DEBUG" OFF)

set(MODEL_SOURCE_DIR .)
include_directories("../..")

############################################################################
########################### CentrEEG #######################################
############################################################################

set(SOURCES1 ${MODEL_SOURCE_DIR}/
        ../models/rnn_learner.cc
        ../models/rnn_learner.hh
        CentrEEG.cc)

add_executable(CentrEEG ${SOURCES1})
target_link_libraries(CentrEEG
        jsoncpp
        ${MLPACK_LIBRARIES}
        ${ARMADILLO_LIBRARIES}
        ${Boost_LIBRARIES})

############################################################################
############################## GMNetEEG ####################################
############################################################################

set(SOURCES2 ${MODEL_SOURCE_DIR}/
        ../models/rnn_learner.cc
        ../models/rnn_learner.hh
        ../networks/dds/binc.hh
        ../networks/dds/dds.hh
        ../networks/dds/dds.cc
        ../networks/dds/dsarch.hh
        ../networks/dds/dsarch.cc
        ../networks/dds/dsarch_tests.hh
        ../networks/gm_network.hh
        ../networks/gm_network.cc
        ../networks/gm_protocol.cc
        ../networks/gm_protocol.hh
        ../networks/feeders.cc
        ../networks/feeders.hh
        GMNetEEG.cc)

add_executable(GMNetEEG ${SOURCES2})
target_link_libraries(GMNetEEG
        jsoncpp
        ${MLPACK_LIBRARIES}
        ${ARMADILLO_LIBRARIES}
        ${Boost_LIBRARIES})


############################################################################
########################## Other Settings ##################################
############################################################################

if (DEBUG)
    message("Compilation with debug info (with ggdb3 flag)")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb3")
else ()
    message("Compilation without debug info (without ggdb3 flag)")
endif ()

#Enable OpenMP package
find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif ()

