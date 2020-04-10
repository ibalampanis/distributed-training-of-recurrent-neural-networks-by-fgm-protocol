#include <utility>
#include <mlpack/core.hpp>
#include <jsoncpp/json/json.h>
#include <iostream>
#include <random>
#include "controller.hh"


using namespace gm_protocol;
using namespace gm_network;
using namespace arma;
using namespace dds;
using namespace controller;

/*********************************************
	Net Container
*********************************************/
template<typename networkType>
void NetContainer<networkType>::Join(networkType *net) { this->push_back(net); }

template<typename networkType>
void NetContainer<networkType>::Leave(int i) { this->erase(this->begin() + i); }


/*********************************************
	Query Container
*********************************************/
void QueryContainer::Join(Query *qry) { this->push_back(qry); }

void QueryContainer::Leave(int i) { this->erase(this->begin() + i); }

/*********************************************
	Progress Bar
*********************************************/
LoopProgressBar::LoopProgressBar(size_t iters) : progress(0),
                                                 nCycles(iters),
                                                 lastPerc(0),
                                                 bUpdateIsCalled(false) {}

void LoopProgressBar::Update() {

    if (!bUpdateIsCalled)
        cout << "0%";

    bUpdateIsCalled = true;

    auto perc = size_t(progress * 100 / (nCycles - 1));

    if (perc < lastPerc)
        return;

    // Update percentage each unit
    if (perc == lastPerc + 1) {
        // erase the correct  number of characters
        if (perc <= 10)
            cout << "\b\b" << perc << '%';
        else if (perc > 10 and perc < 100)
            cout << "\b\b\b" << perc << '%';
        else if (perc == 100)
            cout << "\b\b\b" << perc << '%';
    }

    lastPerc = perc;
    progress++;
    cout << flush;
}


/*********************************************
	Controller
*********************************************/
template<typename networkType>
Controller<networkType>::Controller(string cfg) : configFile(move(cfg)) {

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

template<typename networkType>
void Controller<networkType>::InitializeSimulation() {

    cout << "\n[+]Initializing the star network ..." << endl;
    try {
        Json::Value root;
        ifstream cfgfile(configFile); // Parse from JSON file.
        cfgfile >> root;

        source_id count = 0;

        string netName = root["simulations"].get("net_name", "NoNet").asString();
        string netType = root["simulations"].get("net_type", "NoType").asString();
        string learningAlgorithm = root["net"].get("learning_algorithm",
                                                   "NoAlgo").asString();

        auto query = new gm_protocol::Query(configFile, netName);
        AddQuery(query);

        source_id numOfNodes = (source_id) root["net"].get("number_of_local_nodes",
                                                           1).asInt64();
        set<source_id> nodeIDs;
        for (source_id j = 1; j <= numOfNodes; j++) {
            nodeIDs.insert(count + j);
        }
        // We add one more because of the coordinator.
        count += numOfNodes + 1;

        cout << "\t[+]Initializing RNNs ...";
        try {

//            if(netType == "gm")
//                auto net = new GmNet(nodeIDs, netName, query);
//            else if (netType == "fgm")
//                auto net = new FgmNet(nodeIDs, netName, query);
//            else
//                assert(false);
            auto net = new GmNet(nodeIDs, netName, query);
            AddNet(net);
            net->hub->SetupConnections();
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

template<typename networkType>
void Controller<networkType>::ShowNetworkInfo() const {

    auto *net = _netContainer.front();

    cout << "\n[+]Printing network information ..." << endl;
    cout << "\t-- Network Name: " << net->name() << endl;
    cout << "\t-- Coordinator: " << net->hub->name() << " with ID: " << net->hub->addr() << endl;
    cout << "\t-- Number of nodes: " << net->sites.size() << endl;
    for (size_t j = 0; j < net->sites.size(); j++)
        cout << "\t\t-- Node: " << net->sites.at(j)->name() << " with networkID: " << net->sites.at(j)->site_id()
             << endl;
}

template<typename networkType>
void Controller<networkType>::TrainOverNetwork() {

    cout << "\n[+]Preparing data ..." << endl;
    try {
        DataPreparation();
        cout << "[+]Preparing data ... OK." << endl;
    } catch (...) {
        cout << "[-]Preparing data ... ERROR." << endl;
    }

    cout << "\n[+]Training ... ";
    try {
        auto *net = _netContainer.front();

        net->StartTraining();

        // Using random lib (C++11 Standard) to generate random numbers in the range of the size of sites.
        random_device rd;     // only used once to initialise (seed) engine
        mt19937 rng(rd());    // random-number engine used (Mersenne-Twister in this case)
        uniform_int_distribution<int> uni(0, net->sites.size() - 1); // guaranteed unbiased

        int iterations = trainX.n_cols;
        LoopProgressBar progressBar(iterations);

        // In this loop, whole dataset will get crossed as well as each
        // time point, a random node will get fit by a new point of dataset.
        for (size_t i = 0; i < trainX.n_cols; i++) {
            size_t currentNode = uni(rng);
            arma::cube x = trainX.subcube(arma::span(), arma::span(i, i), arma::span());
            arma::cube y = trainY.subcube(arma::span(), arma::span(i, i), arma::span());

            net->TrainNode(currentNode, x, y);

            progressBar.Update();
        }

        cout << " ... FINISHED." << endl;

        net->ShowTrainingStats();

    } catch (...) {
        cout << " ... ERROR." << endl;
    }
}

template<typename networkType>
void Controller<networkType>::AddNet(networkType *net) { _netContainer.Join(net); }

template<typename networkType>
void Controller<networkType>::AddQuery(Query *qry) { _queryContainer.Join(qry); }

template<typename networkType>
void Controller<networkType>::CreateTimeSeriesData(arma::mat dataset, arma::cube &X, arma::cube &y) {

    for (size_t i = 0; i < dataset.n_cols - rho; i++) {
        X.subcube(arma::span(), arma::span(i), arma::span()) = dataset.submat(arma::span(), arma::span(i, i + rho - 1));
        y.subcube(arma::span(), arma::span(i), arma::span()) = dataset.submat(
                arma::span(dataset.n_rows - 1, dataset.n_rows - 1), arma::span(i + 1, i + rho));
    }
}

template<typename networkType>
void Controller<networkType>::DataPreparation() {

    arma::mat dataset;
    // In Armadillo rows represent features, columns represent data points.
    cout << "\t[+]Reading dataset ...";

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

    cout << "\t[+]Creating time-series data ...";
    try {
        CreateTimeSeriesData(dataset, X, y);
        cout << " OK." << endl;
    } catch (...) {
        cout << " ERROR." << endl;
    }

    cout << "\t[+]Splitting the data into training and testing sets ...";
    try {
        // Split the data into training and testing sets.
        size_t trainingSize = (1 - trainTestRatio) * X.n_cols;
        trainX = X.subcube(arma::span(), arma::span(0, trainingSize - 1), arma::span());
        trainY = y.subcube(arma::span(), arma::span(0, trainingSize - 1), arma::span());
        testX = X.subcube(arma::span(), arma::span(trainingSize, X.n_cols - 1), arma::span());
        testY = y.subcube(arma::span(), arma::span(trainingSize, X.n_cols - 1), arma::span());


        auto *net = _netContainer.front();
        net->hub->testX = testX;
        net->hub->testY = testY;
        net->hub->trainPoints = trainX.n_cols;
        cout << " OK." << endl;
    } catch (...) {
        cout << " ERROR." << endl;
    }
}
