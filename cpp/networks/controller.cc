#include <utility>
#include <mlpack/core.hpp>
#include <jsoncpp/json/json.h>
#include <iostream>
#include "controller.hh"

using namespace gm_protocol;
using namespace gm_network;
using namespace arma;
using namespace dds;
using namespace controller;

using std::vector;
using std::string;

/*********************************************
	Net Container
*********************************************/
template<typename distrNetType>
void NetContainer<distrNetType>::Join(distrNetType *net) { this->push_back(net); }

template<typename distrNetType>
void NetContainer<distrNetType>::Leave(int i) { this->erase(this->begin() + i); }


/*********************************************
	Query Container
*********************************************/
void QueryContainer::Join(Query *qry) { this->push_back(qry); }

void QueryContainer::Leave(int i) { this->erase(this->begin() + i); }


/*********************************************
	Controller
*********************************************/
template<typename distrNetType>
Controller<distrNetType>::Controller(string cfg) : configFile(move(cfg)) {

    Json::Value root;
    ifstream cfgfile(configFile); // Parse from JSON file.
    cfgfile >> root;

    long int randomSeed = (long int) root["simulations"].get("seed", 0).asInt64();
    if (randomSeed >= 0)
        seed = randomSeed;
    else
        seed = time(&randomSeed);

    srand(seed);

    datasetPath = root["data"].get("path", "").asString();
    trainTestRatio = root["hyperparameters"].get("trainTestRatio", -1).asDouble();
    inputSize = root["data"].get("input_size", -1).asInt();
    outputSize = root["data"].get("output_size", -1).asInt();
    rho = root["hyperparameters"].get("rho", -1).asInt();
}

template<typename distrNetType>
void Controller<distrNetType>::InitializeSimulation() {

    cout << "\n[+]Initializing the star network ..." << endl;
    try {
        Json::Value root;
        std::ifstream cfgfile(configFile); // Parse from JSON file.
        cfgfile >> root;

        source_id count = 0;

        string netName = root["simulations"].get("net_name", "NoNet").asString();
        string learningAlgorithm = root["gm_net"].get("learning_algorithm",
                                                      "NoAlgo").asString();

        auto query = new gm_protocol::Query(configFile, netName);
        AddQuery(query);

        source_id numOfNodes = (source_id) root["gm_net"].get("number_of_local_nodes",
                                                              1).asInt64();
        set<source_id> nodeIDs;
        for (source_id j = 1; j <= numOfNodes; j++) {
            nodeIDs.insert(count + j);
        }
        // We add one more because of the coordinator.
        count += numOfNodes + 1;

        cout << "\t[+]Initializing RNNs ...";
        try {
            auto net = new GmNet(nodeIDs, netName, query);
            AddNet(net);
            cout << "\t[+]Initializing RNNs ... OK.";
        } catch (...) {
            cout << "\t[-]Initializing RNNs ... ERROR.";
        }
        msgs = 0;
        bts = 0;
        cout << "\n[+]Initializing the star network ... OK." << endl;
    } catch (...) {
        cout << "\n[-]Initializing the star network ... ERROR." << endl;
    }
}

template<typename distrNetType>
void Controller<distrNetType>::ShowNetworkInfo() const {

    auto *net = _netContainer.front();

    cout << "\n[+]Printing network information ..." << endl;
    cout << "\t-- Network Name: " << net->name() << endl;
    cout << "\t-- Coordinator: " << net->hub->name() << " with ID: " << net->hub->addr() << endl;
    cout << "\t-- Number of nodes: " << net->sites.size() << endl;
    for (size_t j = 0; j < net->sites.size(); j++)
        cout << "\t\t-- Node: " << net->sites.at(j)->name() << " with ID: " << net->sites.at(j)->site_id() << endl;
}

template<typename distrNetType>
void Controller<distrNetType>::TrainOverNetwork() {
    cout << "\n[+]Preparing data ...";
    try {
        DataPreparation();
        cout << "[+]Preparing data ... OK." << endl;
    } catch (...) {
        cout << "[-]Preparing data ... ERROR." << endl;
    }

    cout << "\n[+]Training ...";
    try {
        auto *net = _netContainer.front();

        net->StartTraining();

        // In this loop, whole dataset will get crossed as well as each
        // time point, a random node will get fit by a new point of dataset.
        for (size_t i = 0; i < trainX.n_cols; i++) {
            size_t currentNode = rand() % (net->sites.size());
            arma::cube x = trainX.subcube(arma::span(), arma::span(i, i), arma::span());
            arma::cube y = trainY.subcube(arma::span(), arma::span(i, i), arma::span());

            net->TrainNode(currentNode, x, y);
        }

        net->FinalizeTraining();

        cout << "\n[+]Training ... OK.";
    } catch (...) {
        cout << "\n[+]Training ... ERROR.";
    }
}

template<typename distrNetType>
void Controller<distrNetType>::AddNet(distrNetType *net) { _netContainer.Join(net); }

template<typename distrNetType>
void Controller<distrNetType>::AddQuery(Query *qry) { _queryContainer.Join(qry); }

template<typename distrNetType>
void Controller<distrNetType>::CreateTimeSeriesData(arma::mat dataset, arma::cube &X, arma::cube &y) {
    for (size_t i = 0; i < dataset.n_cols - rho; i++) {
        X.subcube(arma::span(), arma::span(i), arma::span()) = dataset.submat(arma::span(), arma::span(i, i + rho - 1));
        y.subcube(arma::span(), arma::span(i), arma::span()) = dataset.submat(
                arma::span(dataset.n_rows - 1, dataset.n_rows - 1), arma::span(i + 1, i + rho));
    }
}

template<typename distrNetType>
void Controller<distrNetType>::DataPreparation() {

    arma::mat dataset;
    // In Armadillo rows represent features, columns represent data points.
    cout << "\n\t[+]Reading dataset ...";

    try {
        data::Load(datasetPath, dataset, true);
        cout << " OK." << endl;
    } catch (...) {
        cout << " ERROR." << endl;
    }

    // Scale all data into the range (0, 1) for increased numerical stability.
    data::MinMaxScaler scale;
    scale.Fit(dataset);
    scale.Transform(dataset, dataset);


    // We need to represent the input data for RNN in an arma::cube (3D matrix).
    // The 3rd dimension is rho, the number of past data records the RNN uses for learning.
    arma::cube X, y;
    X.set_size(inputSize, dataset.n_cols - rho + 1, rho);
    y.set_size(outputSize, dataset.n_cols - rho + 1, rho);


    CreateTimeSeriesData(dataset, X, y);

    // Split the data into training and testing sets.
    size_t trainingSize = (1 - trainTestRatio) * X.n_cols;
    trainX = X.subcube(arma::span(), arma::span(0, trainingSize - 1), arma::span());
    trainY = y.subcube(arma::span(), arma::span(0, trainingSize - 1), arma::span());
    testX = X.subcube(arma::span(), arma::span(trainingSize, X.n_cols - 1), arma::span());
    testY = y.subcube(arma::span(), arma::span(trainingSize, X.n_cols - 1), arma::span());

    auto *net = _netContainer.front();

    // Define sizes
    size_t chunks = net->size() - 1; // The coordinator mustn't be taken into account
    size_t trainXSize = trainX.n_cols;
    size_t chunkSizeX = trainXSize / chunks;
//    size_t moduloX = trainXSize % chunks;
    size_t trainYSize = trainY.n_cols;
    size_t chunkSizeY = trainYSize / chunks;
//    size_t moduloY = trainYSize % chunks;

    cout << "\t[+]Sliver dataset to " << chunks << " chunks ...";
    try {
        for (size_t i = 0; i < net->sites.size(); i++) {
            // Deliver trainX to each node
            size_t start = i * chunkSizeX;
            size_t end = (i + 1) * chunkSizeX;
            net->sites.at(i)->trainX = trainX.subcube(arma::span(), arma::span(start, end - 1), arma::span());

            // Deliver trainY to each node
            start = i * chunkSizeY;
            end = (i + 1) * chunkSizeY;
            net->sites.at(i)->trainY = trainY.subcube(arma::span(), arma::span(start, end - 1), arma::span());

            // All nodes have the same testSet
            net->sites.at(i)->testX = testX;
            net->sites.at(i)->testY = testY;
        }
        cout << " OK." << endl;
    } catch (...) {
        cout << " ERROR." << endl;
    }

}