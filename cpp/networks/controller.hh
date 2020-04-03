#ifndef DISTRIBUTED_TRAINING_OF_RECURRENT_NEURAL_NETWORKS_BY_FGM_PROTOCOL_CONTROLLER_HH
#define DISTRIBUTED_TRAINING_OF_RECURRENT_NEURAL_NETWORKS_BY_FGM_PROTOCOL_CONTROLLER_HH


#include <fstream>
#include <iostream>
#include <jsoncpp/json/json.h>
#include <mlpack/core.hpp>
#include "gm_protocol.hh"
#include "gm_network.hh"
#include "ddsim/dds.hh"


namespace controller {

    using namespace std;
    using namespace gm_protocol;
    using namespace gm_network;
    using namespace arma;
    using namespace dds;
    using std::vector;
    using std::string;

    /** A Vector container for the networks. */
    template<typename distrNetType>
    class NetContainer : public vector<distrNetType *> {

    public:
        using vector<distrNetType *>::vector;

        void Join(distrNetType *net);

        void Leave(int i);

    };

    /** A Vector container for the queries. */
    class QueryContainer : public vector<Query *> {

    public:
        using vector<Query *>::vector;

        void Join(Query *qry);

        void Leave(int i);

    };

    /**
     * The purpose of Controller class is to synchronize the training of the
     * network nodes by providing the appropriate data points to these.
     */
    template<typename distrNetType>
    class Controller {

    protected:
        std::string configFile;                     // JSON file to read the hyperparameters.
        time_t seed;                                // The seed for the random generator.
        size_t numberOfFeatures;                    // The number of features of each datapoint.
        NetContainer<distrNetType> _netContainer;   // A container for networks.
        QueryContainer _queryContainer;             // A container for queries.

        // Stats
        vector<chan_frame> stats;
        vector<vector<vector<size_t>>> differentialCommunication;
        size_t msgs{};
        size_t bts{};
        vector<vector<double>> differentialAccuracy;
        bool logDiffAcc;

    public:

        /** Constructor */
        explicit Controller<distrNetType>(string cfg);

        /** This method initializes all the networks. */
        void InitializeSimulation();

        /** This method prints the star learning network for debbuging purposes. */
        void ShowNetworkInfo() const;

        /** This method handles communication information after each batch. */
        void HandleDifferentialInfo();

        void TrainOverNetwork();

        /** This method appends a network in the network container. */
        void AddNet(distrNetType *net);

        /** This method appends a query in the query container. */
        void AddQuery(Query *qry);

        size_t RandomInt(size_t maxValue);

        size_t NumberOfFeatures();

    };

} // end namespace controller

#endif //DISTRIBUTED_TRAINING_OF_RECURRENT_NEURAL_NETWORKS_BY_FGM_PROTOCOL_CONTROLLER_HH
