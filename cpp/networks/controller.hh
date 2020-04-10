#ifndef DISTRIBUTED_TRAINING_OF_RECURRENT_NEURAL_NETWORKS_BY_FGM_PROTOCOL_CONTROLLER_HH
#define DISTRIBUTED_TRAINING_OF_RECURRENT_NEURAL_NETWORKS_BY_FGM_PROTOCOL_CONTROLLER_HH


#include <fstream>
#include <iostream>
#include <jsoncpp/json/json.h>
#include <mlpack/core.hpp>
#include "protocols.hh"
#include "gm.hh"
#include "ddsim/dds.hh"


namespace controller {

    using namespace std;
    using namespace gm_protocol;
    using namespace gm_network;
    using namespace arma;
    using namespace dds;

    // A Vector container for the networks. 
    template<typename networkType>
    class NetContainer : public vector<networkType *> {

    public:
        using vector<networkType *>::vector;

        void Join(networkType *net);

        void Leave(int i);

    };

    // A Vector container for the queries. 
    class QueryContainer : public vector<Query *> {

    public:
        using vector<Query *>::vector;

        void Join(Query *qry);

        void Leave(int i);

    };

    // A very simple progress bar for loops (based on this repo: https://github.com/gipert/progressbar)
    class LoopProgressBar {

    public:
        explicit LoopProgressBar(size_t iters);

        void Update();

    private:
        size_t progress;
        size_t nCycles;
        size_t lastPerc;
        bool bUpdateIsCalled;
    };

    // The purpose of Controller class is to synchronize the training of the
    // network nodes by providing the appropriate data points to these.
    template<typename networkType>
    class Controller {

    protected:
        string configFile;                     // JSON file to read the hyperparameters.
        time_t seed;                                // The seed for the random generator.
        NetContainer<networkType> _netContainer;   // A container for networks.
        QueryContainer _queryContainer;             // A container for queries.

        // Dataset and model parameters 
        arma::cube trainX, trainY;                  // Trainset data points and labels
        arma::cube testX, testY;                    // Testset data points and labels
        size_t inputSize;                           // Number of neurons at the input layer
        size_t outputSize;                          // Number of neurons at the output layer
        string datasetPath;                         // Path for finding dataset file
        double trainTestRatio;                      // Testing data is taken from the dataset in this ratio
        size_t rho;                                 // Number of time steps to look backward for in the RNN

        // Stats
        vector<chan_frame> stats;
        size_t msgs{};
        size_t bts{};


    public:

        // Constructor 
        explicit Controller<networkType>(string cfg);

        // This method initializes all the networks. 
        void InitializeSimulation();

        // This method prints the star learning network for debbuging purposes. 
        void ShowNetworkInfo() const;

        void TrainOverNetwork();

        // This method appends a network in the network container. 
        void AddNet(networkType *net);

        // This method appends a query in the query container. 
        void AddQuery(Query *qry);

        void CreateTimeSeriesData(arma::mat dataset, arma::cube &X, arma::cube &y);

        void DataPreparation();
    };

} // end namespace controller

#endif //DISTRIBUTED_TRAINING_OF_RECURRENT_NEURAL_NETWORKS_BY_FGM_PROTOCOL_CONTROLLER_HH
