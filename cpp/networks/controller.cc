#include <utility>

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

    // Boolean flag to determine if the experiment will log the differential accuracies.
    logDiffAcc = root["simulations"].get("log_diff_acc", false).asBool();

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

        auto net = new GmNet(nodeIDs, netName, query);
        AddNet(net);

        cout << "\t[+]Initializing RNNs ... OK.";

        // Initializing the differential communication statistics.
        stats.push_back(chan_frame(net));
        vector<vector<size_t>> dif_com;
        vector<size_t> dif_msgs;
        vector<size_t> dif_bts;
        dif_msgs.push_back(0);
        dif_bts.push_back(0);
        dif_com.push_back(dif_msgs);
        dif_com.push_back(dif_bts);
        differentialCommunication.push_back(dif_com);
        vector<double> dif_acc;
        dif_acc.push_back(0.);
        differentialAccuracy.push_back(dif_acc);

        msgs = 0;
        bts = 0;
        cout << "\n[+]Initializing the star network ... OK." << endl;
    } catch (...) {
        cout << "\n[+]Initializing the star network ... ERROR" << endl;
    }

}

template<typename distrNetType>
void Controller<distrNetType>::ShowNetworkInfo() const {
    cout << "\n[+]Printing information about network ..." << endl;
    cout << "\t-- Number of networks: " << _netContainer.size() << endl;
    for (auto net:_netContainer) {
        cout << "\t-- Network Name: " << net->name() << endl;
        cout << "\t-- Number of nodes: " << net->sites.size() << endl;
        cout << "\t-- Coordinator " << net->hub->name() << " with local address: " << net->hub->addr() << endl;
        for (size_t j = 0; j < net->sites.size(); j++) {
            cout << "\t-- Node (" << j + 1 << "/" << net->sites.size() << ") " << net->sites.at(j)->name()
                 << " with local address: " << net->sites.at(j)->site_id() << endl;
        }
    }
}
// PENDING implement HandleDifferentialInfo() if is needed
//template<typename distrNetType>
//void Controller<distrNetType>::HandleDifferentialInfo() {}
// PENDING: implement TrainOverNetwork()
template<typename distrNetType>
void Controller<distrNetType>::TrainOverNetwork() {}

template<typename distrNetType>
void Controller<distrNetType>::AddNet(distrNetType *net) { _netContainer.Join(net); }

template<typename distrNetType>
void Controller<distrNetType>::AddQuery(Query *qry) { _queryContainer.Join(qry); }

template<typename distrNetType>
size_t Controller<distrNetType>::RandomInt(size_t maxValue) { return std::rand() % maxValue; }

template<typename distrNetType>
size_t Controller<distrNetType>::NumberOfFeatures() { return numberOfFeatures; }
