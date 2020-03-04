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
    class NetContainer : public std::vector<distrNetType *> {
    public:
        using std::vector<distrNetType *>::vector;

        inline void join(distrNetType *net) { this->push_back(net); }

        inline void leave(int const i) { this->erase(this->begin() + i); }

    };

    // A vector container for the queries.
    class QueryContainer : public vector<continuous_query *> {
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
    class Feeder {
    protected:
        std::string config_file; // JSON file to read the hyperparameters.
        time_t seed; // The seed for the random generator.
        size_t batchSize{}; // The batch learning size.
        size_t warmupSize{}; // The size of the warmup dataset.
        arma::mat testSet; // Test dataset for validation of the classification algorithm.
        arma::mat testResponses; // Arma row vector containing the labels of the evaluation set.

        size_t test_size{}; // Size of test dataset. [optional]
        bool negative_labels; // If true it sets 0 labels to -1. [optional]

        NetContainer<distrNetType> _net_container; // A container for networks.
        QueryContainer _query_container; // A container for queries.

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

        Feeder(string cfg);

        /**
            Method that creates the test dataset.
            This method passes one time through the entire dataset,
            if the dataset is stored in a hdf5 file.
        **/
        virtual void MakeTestDataset() {}

        /* Method that puts a network in the network container. */
        void AddNet(distrNetType *net) { _net_container.join(net); }

        /* Method that puts a network in the network container. */
        void AddQuery(continuous_query *qry) { _query_container.join(qry); }

        /* Method initializing all the networks. */
        void InitializeSimulation();

        /* Method that prints the star learning network for debbuging purposes. */
        void PrintStarNets() const;

        /* Method that gathers communication info after each streaming batch. */
        void GatherDifferentialInfo();

        // Getters.
        inline arma::mat &GetTestSet() { return testSet; }

        inline arma::mat &GetTestSetLabels() { return testResponses; }

        inline arma::mat *GetPTestSet() { return &testSet; }

        inline arma::mat *GetPTestSetLabels() { return &testResponses; }

        inline size_t GetRandomInt(size_t maxValue) { return std::rand() % maxValue; }

        inline size_t GetNumberOfFeatures() { return 0; }

        virtual void GetStatistics() {}
    };

    template<typename distrNetType>
    class RandomFeeder : public Feeder<distrNetType> {
    protected:
        size_t test_size; // Starting test data point.
        size_t number_of_features; // The number of features of each datapoint.
        bool linearly_seperable; // Determines if the random dataset is linearly seperable.
        arma::mat target; // The moving target disjunction of the stream.
        size_t targets; // The total number of changed targets.

        size_t numOfPoints = 12800000; // Total number of datapoints.
        size_t numOfMaxRounds = 100000; // Maximum number of monitored rounds.

    public:

        explicit RandomFeeder(const string &cfg);

        void MakeTestDataset() override;

        void GenNewTarget();

        arma::dvec GenPoint();

        void TrainNetworks();

        void Train(arma::mat &batch, arma::mat &labels);

        virtual void GetStatistics() {}

        virtual size_t GetNumberOfFeatures() override { return number_of_features; }
    };

}

#endif //DISTRIBUTED_TRAINING_OF_RECURRENT_NEURAL_NETWORKS_BY_FGM_PROTOCOL_FEEDERS_HH
