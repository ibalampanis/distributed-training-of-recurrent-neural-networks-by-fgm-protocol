cmake_minimum_required(VERSION 2.6)
project(Distributed_Training_of_Recurrent_Neural_Networks_by_FGM_protocol)
set(DCMAKE_CXX_COMPILER "g++-7")
set(CMAKE_CXX_STANDARD 14)  # enable C++14 standard
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

option(DEBUG "DEBUG" ON)

set(CPP_SOURCE_DIR ..)
include_directories("../..")
include_directories(${PYTHON_INCLUDE_DIRS})
#set(DDS_SOURCES
#        "/home/ibalampanis/Documents/vsam/source-codes/ddssim/cpp/hdv.cc"
#        /home/ibalampanis/Documents/vsam/source-codes/ddssim/cpp/ddsim.cc
#        /home/ibalampanis/Documents/vsam/source-codes/ddssim/cpp/output.cc
#        /home/ibalampanis/Documents/vsam/source-codes/ddssim/cpp/eca.cc
#        /home/ibalampanis/Documents/vsam/source-codes/ddssim/cpp/agms.cc
#        /home/ibalampanis/Documents/vsam/source-codes/ddssim/cpp/data_source.cc
#        /home/ibalampanis/Documents/vsam/source-codes/ddssim/cpp/method.cc
#        /home/ibalampanis/Documents/vsam/source-codes/ddssim/cpp/cfgfile.cc
#        /home/ibalampanis/Documents/vsam/source-codes/ddssim/cpp/dsarch.cc
#        /home/ibalampanis/Documents/vsam/source-codes/ddssim/cpp/accurate.cc
#        /home/ibalampanis/Documents/vsam/source-codes/ddssim/cpp/query.cc
#        /home/ibalampanis/Documents/vsam/source-codes/ddssim/cpp/results.cc
#        /home/ibalampanis/Documents/vsam/source-codes/ddssim/cpp/sz_quorum.cc
#        /home/ibalampanis/Documents/vsam/source-codes/ddssim/cpp/sz_bilinear.cc
#        /home/ibalampanis/Documents/vsam/source-codes/ddssim/cpp/tods.cc
#        /home/ibalampanis/Documents/vsam/source-codes/ddssim/cpp/safezone.cc
#        /home/ibalampanis/Documents/vsam/source-codes/ddssim/cpp/gm_proto.cc
#        /home/ibalampanis/Documents/vsam/source-codes/ddssim/cpp/gm_szone.cc
#        /home/ibalampanis/Documents/vsam/source-codes/ddssim/cpp/gm_query.cc
#        /home/ibalampanis/Documents/vsam/source-codes/ddssim/cpp/fgm.cc
#        /home/ibalampanis/Documents/vsam/source-codes/ddssim/cpp/sgm.cc
#        /home/ibalampanis/Documents/vsam/source-codes/ddssim/cpp/frgm.cc
#        )


############################################################################
########################### CentrEEG #######################################
############################################################################

set(MLSOURCES ../models/rnn_learner.cc)

add_executable(CentrEEG centr_eeg.cc ${MLSOURCES})
target_link_libraries(
        CentrEEG
        jsoncpp
        ${MLPACK_LIBRARIES}
        ${ARMADILLO_LIBRARIES}
        ${Boost_LIBRARIES}
)

############################################################################
############################## GMNetEEG ####################################
############################################################################

file(GLOB DDSSIM_SOURCES ../networks/ddsim/*.cc)
file(GLOB NET_SOURCES ../networks/*.cc)
set(ML_SOURCES ../models/rnn_learner.cc)
add_library(DML_LIBRARIES ${ML_SOURCES} ${DDSSIM_SOURCES} ${NET_SOURCES})
install(TARGETS DML_LIBRARIES DESTINATION lib)


add_executable(GMNetEEG gm_net_eeg.cc ${DML_LIBRARIES})
target_link_libraries(GMNetEEG
        jsoncpp
        ${HDF5_LIBRARIES}
        ${MLPACK_LIBRARIES}
        ${ARMADILLO_LIBRARIES}
        ${Boost_LIBRARIES}
        ${PYTHON_LIBRARIES}
        ${DML_LIBRARIES}
        )


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

