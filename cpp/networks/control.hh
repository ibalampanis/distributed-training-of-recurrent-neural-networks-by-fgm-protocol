#ifndef DISTRIBUTED_TRAINING_OF_RECURRENT_NEURAL_NETWORKS_BY_FGM_PROTOCOL_SUPERVISOR_HH
#define DISTRIBUTED_TRAINING_OF_RECURRENT_NEURAL_NETWORKS_BY_FGM_PROTOCOL_SUPERVISOR_HH


#include <fstream>
#include <iostream>
#include <jsoncpp/json/json.h>
#include <mlpack/core.hpp>
#include "protocols.hh"
#include "gm.hh"
#include "fgm.hh"
#include "ddsim/dds.hh"


namespace controller {

    using namespace std;
    using namespace protocols;
    using namespace algorithms;
    using namespace arma;
    using namespace dds;


    // The purpose of Controller class is to synchronize the training of the
    // network nodes by providing the appropriate data points to these.
    template<typename networkType>
    class Controller {

    protected:
        string configFile;                          // JSON file to read the hyperparameters.

        // Dataset and model parameters 
        arma::cube trainX, trainY;                  // Trainset data points and labels
        arma::cube testX, testY;                    // Testset data points and labels
        size_t inputSize;                           // Number of neurons at the input layer
        size_t outputSize;                          // Number of neurons at the output layer
        string datasetPath;                         // Path for finding dataset file
        string datasetName;                         // The name of the dataset we use
        double trainTestRatio;                      // Testing data is taken from the dataset in this ratio
        size_t rho;                                 // Number of time steps to look backward for in the RNN
        bool warmup;                                // Define if hub warmup is needed
        size_t miniBatchSize;                       // Number of training samples given each time
        bool interStats;                            // Define if intermediate communication stats is needed

        // Stats
        chan_frame stats;
        size_t msgs{};
        size_t bts{};
        networkType *net;


    public:
        // Constructor 
        explicit Controller<networkType>(string cfg);

        // This method initializes all the networks. 
        void InitializeSimulation();

        // This method prints the star learning network for debbuging purposes. 
        void ShowNetworkInfo() const;

        void CreateTimeSeriesData(arma::mat dataset, arma::cube &X, arma::cube &y);

        void DataPreparation();

        void TrainOverNetwork();

        // Method that monitors communication stattistics after each mini-batch of training samples.
        void GatherIntermediateNetStats();

        void ShowNetworkStats();
    };

    // A very simple progress bar for loops (based on this repo: https://github.com/gipert/progressbar)
    class LoopProgressPercentage {

    public:
        explicit LoopProgressPercentage(size_t iters);

        void Update();

    private:
        size_t progress;
        size_t nCycles;
        size_t lastPerc;
        bool bUpdateIsCalled;
    };

} // end namespace controller

#endif //DISTRIBUTED_TRAINING_OF_RECURRENT_NEURAL_NETWORKS_BY_FGM_PROTOCOL_SUPERVISOR_HH
