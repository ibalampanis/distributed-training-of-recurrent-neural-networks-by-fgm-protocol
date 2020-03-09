#ifndef DISTRIBUTED_TRAINING_OF_RECURRENT_NEURAL_NETWORKS_BY_FGM_PROTOCOL_CONTROLLER_HH
#define DISTRIBUTED_TRAINING_OF_RECURRENT_NEURAL_NETWORKS_BY_FGM_PROTOCOL_CONTROLLER_HH


#include <fstream>
#include <iostream>
#include <jsoncpp/json/json.h>
#include <mlpack/core.hpp>
#include "gm_protocol.hh"
#include "gm_network.hh"
#include "dds/dds.hh"


namespace controller {

    using namespace gm_protocol;
    using namespace arma;
    using namespace dds;
    using std::vector;
    using std::string;

    /**
     * Vector container for the networks.
     */
    template<typename distrNetType>
    class NetContainer : public std::vector<distrNetType *> {

    public:
        using std::vector<distrNetType *>::vector;

        void Join(distrNetType *net);

        void Leave(int i);

    };

    /**
     * Vector container for the queries.
     */
    class QueryContainer : public vector<Query *> {

    public:
        using vector<Query *>::vector;

        void Join(Query *qry);

        void Leave(int i);

    };

    /**
     * The purpose of Controller class is to synchronize the testing of the networks
     * by providing the appropriate data points to the nodes of each net.
     */
    template<typename distrNetType>
    class Controller {
    protected:
        std::string configFile;                     // JSON file to read the hyperparameters.
        time_t seed;                                // The seed for the random generator.
        size_t batchSize{};                         // The batch learning size.
        size_t warmupSize{};                        // The size of the warmup dataset.
        size_t testSize;                            // Starting test data point.
        size_t numberOfFeatures;                    // The number of features of each datapoint.
        arma::mat target;                           // The moving target disjunction of the stream.
        size_t targets;                             // The total number of changed targets.
        size_t numOfPoints = 12800000;              // Total number of datapoints.
        size_t numOfMaxRounds = 100000;             // Maximum number of monitored rounds.


        NetContainer<distrNetType> _netContainer;   // A container for networks.
        QueryContainer _queryContainer;             // A container for queries.

        // Stream Distribution
        bool uniformDistr;
        vector<vector<set<size_t>>> net_dists;
        float Bprob;
        float siteRatio;

        // Statistics collection
        vector<chan_frame> stats;
        vector<vector<vector<size_t>>> differentialCommunication;
        size_t msgs{};
        size_t bts{};
        vector<vector<double>> differentialAccuracy;
        bool logDiffAcc;

    public:

        /** Constructor */
        explicit Controller(const string &cfg);

        /** This method puts a network in the network container */
        void AddNet(distrNetType *net);

        /** This method puts a query in the query container */
        void AddQuery(Query *qry);

        /** This method initializes all the networks */
        void InitializeSimulation();

        /** This method prints the star learning network for debbuging purposes */
        void PrintStarNets() const;

        /** This method gathers communication info after each streaming batch */
        void GatherDifferentialInfo();

        void TrainNetworks();

        void Train(arma::mat &batch, arma::mat &labels);

        /** Getters */
        size_t GetRandomInt(size_t maxValue);

        void GetStatistics() {}

        size_t GetNumberOfFeatures();

    };

}

#endif //DISTRIBUTED_TRAINING_OF_RECURRENT_NEURAL_NETWORKS_BY_FGM_PROTOCOL_CONTROLLER_HH
