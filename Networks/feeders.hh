#ifndef DISTRIBUTED_TRAINING_OF_RECURRENT_NEURAL_NETWORKS_BY_FGM_PROTOCOL_FEEDERS_HH
#define DISTRIBUTED_TRAINING_OF_RECURRENT_NEURAL_NETWORKS_BY_FGM_PROTOCOL_FEEDERS_HH


#include <fstream>
#include <iostream>
#include <jsoncpp/json/json.h>
#include <mlpack/core.hpp>
#include "gm_protocol.hh"
#include "gm_network.hh"
#include "dds/dds.hh"


namespace feeders {

    using namespace gm_protocol;
    using namespace arma;
    using namespace dds;
    using std::vector;
    using std::string;

    // vector container for the networks.
    template<typename distrNetType>
    class net_container : public std::vector<distrNetType *> {
    public:
        using std::vector<distrNetType *>::vector;

        inline void join(distrNetType *net) { this->push_back(net); }

        inline void leave(int const i) { this->erase(this->begin() + i); }

    };

    // A vector container for the queries.
    class query_container : public vector<continuous_query *> {
    public:
        using vector<continuous_query *>::vector;

        inline void join(continuous_query *qry) { this->push_back(qry); }

        inline void leave(int const i) { this->erase(this->begin() + i); }

    };

    /**
	    A feeders purpose is to synchronize the testing of the networks
	    by providing the appropriate data stream to the nodes of each net.
	**/
    template<typename distrNetType>
    class feeder {
    protected:
        std::string config_file; // JSON file to read the hyperparameters.
        time_t seed; // The seed for the random generator.
        size_t batchSize{}; // The batch learning size.
        size_t warmupSize{}; // The size of the warmup dataset.
        arma::mat testSet; // Test dataset for validation of the classification algorithm.
        arma::mat testResponses; // Arma row vector containing the labels of the evaluation set.

        size_t test_size{}; // Size of test dataset. [optional]
        bool negative_labels; // If true it sets 0 labels to -1. [optional]

        net_container<distrNetType> _net_container; // A container for networks.
        query_container _query_container; // A container for queries.

        // Stream Distribution
        bool uniform_distr;
        vector<vector<set<size_t>>> net_dists;
        float B_prob;
        float site_ratio;

        // Statistics collection.
        vector<chan_frame> stats;
        vector<vector<vector<size_t>>> differential_communication;
        size_t msgs{};
        size_t bts{};
        vector<vector<double>> differential_accuracy;
        bool log_diff_acc;

    public:

        feeder(string cfg);

        /**
            Method that creates the test dataset.
            This method passes one time through the entire dataset,
            if the dataset is stored in a hdf5 file.
        **/
        virtual void makeTestDataset() {}

        /* Method that puts a network in the network container. */
        void addNet(distrNetType *net) { _net_container.join(net); }

        /* Method that puts a network in the network container. */
        void addQuery(continuous_query *qry) { _query_container.join(qry); }

        /* Method initializing all the networks. */
        void initializeSimulation();

        /* Method that prints the star learning network for debbuging purposes. */
        void printStarNets() const;

        /* Method that gathers communication info after each streaming batch. */
        void gatherDifferentialInfo();

        // Getters.
        arma::mat &getTestSet() { return testSet; }

        arma::mat &getTestSetLabels() { return testResponses; }

        arma::mat *getPTestSet() { return &testSet; }

        arma::mat *getPTestSetLabels() { return &testResponses; }

        size_t getRandomInt(size_t maxValue) { return std::rand() % maxValue; }

        virtual size_t getNumberOfFeatures() { return 0; }

        virtual void getStatistics() {}
    };

    template<typename distrNetType>
    class Random_Feeder : public feeder<distrNetType> {
    protected:
        size_t test_size; // Starting test data point.
        size_t number_of_features; // The number of features of each datapoint.
        bool linearly_seperable; // Determines if the random dataset is linearly seperable.
        arma::mat target; // The moving target disjunction of the stream.
        size_t targets; // The total number of changed targets.

        size_t numOfPoints = 12800000; // Total number of datapoints.
        size_t numOfMaxRounds = 100000; // Maximum number of monitored rounds.

    public:

        explicit Random_Feeder(const string &cfg);

        void makeTestDataset() override;

        void GenNewTarget();

        arma::dvec GenPoint();

        void TrainNetworks();

        void Train(arma::mat &batch, arma::mat &labels);

        virtual void getStatistics() {}

        inline size_t getNumberOfFeatures() override { return number_of_features; }
    };

}

#endif //DISTRIBUTED_TRAINING_OF_RECURRENT_NEURAL_NETWORKS_BY_FGM_PROTOCOL_FEEDERS_HH
