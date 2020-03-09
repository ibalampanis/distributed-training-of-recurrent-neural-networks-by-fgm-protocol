#include "controller.hh"

using namespace gm_protocol;
using namespace arma;
using namespace dds;
using namespace controller;
using std::vector;
using std::string;

/*********************************************
	NetContainer
*********************************************/
template<typename distrNetType>
void NetContainer<distrNetType>::Join(distrNetType *net) { this->push_back(net); }

template<typename distrNetType>
void NetContainer<distrNetType>::Leave(int i) { this->erase(this->begin() + i); }


/*********************************************
	QueryContainer
*********************************************/
void QueryContainer::Join(Query *qry) { this->push_back(qry); }

void QueryContainer::Leave(int i) { this->erase(this->begin() + i); }


/*********************************************
	Controller
*********************************************/
template<typename distrNetType>
Controller<distrNetType>::Controller(const string &cfg) : configFile(std::move(cfg)) {
    Json::Value root;
    std::ifstream cfgfile(configFile); // Parse from JSON file.
    cfgfile >> root;

    long int randomSeed = (long int) root["simulations"].get("seed", 0).asInt64();
    if (randomSeed >= 0) {
        seed = randomSeed;
    } else {
        seed = time(&randomSeed);
    }
    std::srand(seed);

    // Get the stream distribution.
    uniformDistr = root[root["simulations"].get("stream_distribution", "No_Distribution").asString()]
            .get("uniform", true).asBool();
    if (!uniformDistr) {
        Bprob = root[root["simulations"].get("stream_distribution", "No_Distribution").asString()]
                .get("B_prob", -1.).asFloat();
        if (Bprob > 1. || Bprob < 0.) {
            cout << "Invalid parameter B_prob. Probabilities must be in the interval [0,1]" << endl;
            throw;
        }
        siteRatio = root[root["simulations"].get("stream_distribution", "No_Distribution").asString()]
                .get("site_ratio", -1.).asFloat();
        if (siteRatio > 1. || siteRatio < 0.) {
            cout << "Invalid parameter site_ratio. Ratios must be in the interval [0,1]" << endl;
            throw;
        }
    }

    // Boolean flag to determine if the experiment will log the differential accuracies.
    logDiffAcc = root["simulations"].get("log_diff_acc", false).asBool();

    try {
        Json::Value root;
        std::ifstream cfgfile(this->configFile); // Parse from JSON file.
        cfgfile >> root;

        this->batchSize = root["tests_Generated_Data"].get("batch_size", 1).asInt64();

        numberOfFeatures = root["tests_Generated_Data"].get("number_of_features", 20).asInt64();
        if (numberOfFeatures <= 0) {
            cout << endl << "Incorrect parameter number_of_features" << endl;
            cout << "Acceptable number_of_features parameters are all the positive integers." << endl;
            throw;
        }

        testSize = root["tests_Generated_Data"].get("test_size", 100000).asDouble();
        if (testSize < 0) {
            cout << endl << "Incorrect parameter test_size." << endl;
            cout << "Acceptable test_size parameters are all the positive integers." << endl;
            throw;
        }

        std::srand(this->seed);

        cout << "test size : " << testSize << endl << endl;

        targets = 1;

    } catch (...) {
        cout << endl << "Something went wrong in Controller object construction." << endl;
        throw;
    }

}

template<typename distrNetType>
void Controller<distrNetType>::AddNet(distrNetType *net) { _netContainer.Join(net); }

template<typename distrNetType>
void Controller<distrNetType>::AddQuery(Query *qry) { _queryContainer.Join(qry); }

template<typename distrNetType>
void Controller<distrNetType>::InitializeSimulation() {

    cout << "Initializing the star nets..." << endl << endl << endl;
    Json::Value root;
    std::ifstream cfgfile(configFile); // Parse from JSON file.
    cfgfile >> root;

    source_id count = 0;
    size_t numberOfNets = root["simulations"].get("number_of_networks", 0).asInt();
    for (size_t i = 1; i <= numberOfNets; i++) {

        string netName = root["simulations"].get("net_name_" + std::to_string(i), "NoNet").asString();
        string learningAlgorithm = root["gm_network_" + netName].get("learning_algorithm",
                                                                     "NoAlgo").asString();

        cout << "Initializing the network " << netName << " with " << learningAlgorithm << " learner."
             << endl;

        auto query = new gm_protocol::Query(configFile, netName);
        AddQuery(query);
        cout << "Query added." << endl;


        source_id numOfNodes = (source_id) root["gm_network_" + netName].get("number_of_local_nodes",
                                                                             1).asInt64();
        set<source_id> nodeIDs;
        for (source_id j = 1; j <= numOfNodes; j++) {
            nodeIDs.insert(count + j);
        }
        count += numOfNodes + 1; // We add one because of the coordinator.

        auto net = new GmNet(nodeIDs, netName, _queryContainer.at(i - 1));
        AddNet(net);
        cout << "Net " << netName << " initialized." << endl << endl;
    }
    for (auto net:_netContainer) {
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

        // Initializing the stream distributions of the sites of the network.
        if (!uniformDistr) {
            set<size_t> B;
            set<size_t> B_compl;
            vector<set<size_t>> net_distr;
            for (size_t i = 0; i < net->sites.size(); i++) {
                B_compl.insert(i);
            }
            for (size_t i = 0; i < std::floor(net->sites.size() * siteRatio); i++) {
                size_t n = std::rand() % (net->sites.size());
                while (B.find(n) != B.end()) {
                    n = std::rand() % (net->sites.size());
                }
                B.insert(n);
                B_compl.erase(n);
            }
            net_distr.push_back(B);
            net_distr.push_back(B_compl);
            net_dists.push_back(net_distr);
        }
    }
    msgs = 0;
    bts = 0;
    cout << endl << "networks initialized." << endl;
}

template<typename distrNetType>
void Controller<distrNetType>::PrintStarNets() const {
    cout << endl << "Printing the nets." << endl;
    cout << "Number of networks : " << _netContainer.size() << endl;
    for (auto net:_netContainer) {
        cout << endl;
        cout << "Network Name : " << net->name() << endl;
        cout << "Number of nodes : " << net->sites.size() << endl;
        cout << "Coordinator " << net->hub->name() << " with address ";//<< net->hub->addr() << endl;
        for (size_t j = 0; j < net->sites.size(); j++) {
            cout << "Site " << net->sites.at(j)->name() << " with address " << net->sites.at(j)->site_id()
                 << endl;
        }
    }
}

template<typename distrNetType>
void Controller<distrNetType>::GatherDifferentialInfo() {
    // Gathering the info of the communication triggered by the streaming batch.
    for (size_t i = 0; i < _netContainer.size(); i++) {
        size_t batch_messages = 0;
        size_t batch_bytes = 0;

        for (auto chnl:stats.at(i)) {
            batch_messages += chnl->messages_received();
            batch_bytes += chnl->bytes_received();
        }

        differentialCommunication.at(i).at(0)
                .push_back(batch_messages - msgs);

        differentialCommunication.at(i).at(1)
                .push_back(batch_bytes - bts);

        msgs = batch_messages;
        bts = batch_bytes;

        if (logDiffAcc)
            differentialAccuracy.at(i).push_back(_netContainer.at(i)->hub->getAccuracy());
    }
}

template<typename distrNetType>
void Controller<distrNetType>::TrainNetworks() {
    size_t count = 0; // Count the number of processed elements.


    while (count < numOfPoints) {

        arma::mat point = arma::zeros<arma::mat>(numberOfFeatures, 1);
        arma::mat label = arma::zeros<arma::mat>(1, 1);

        if (arma::dot(point.unsafe_col(0), target.unsafe_col(0)) >= 1.) {
            label(0, 0) = 1.;
        } else {
            if (this->negative_labels) {
                label(0, 0) = -1.;
            }
        }

        // Update the number of processed elements.
        count += 1;

        for (size_t i = 0; i < this->_netContainer.size(); i++) {
            this->_netContainer.at(i)->StartRound(); // Each hub initializes the first round.
        }

        Train(point, label);

        if (count % 5000 == 0) {
            cout << "count : " << count << endl;
        }

    }

    for (auto net:this->_netContainer) {
        net->FinishProcess();
    }
    count = 0;
    cout << "Targets : " << targets << endl;
}

template<typename distrNetType>
void Controller<distrNetType>::Train(arma::mat &point, arma::mat &label) {
    for (auto net:this->_netContainer) {

        size_t random_node = std::rand() % (net->sites.size());

        // Train on data point.
        net->ProcessRecord(random_node, point, label);

    }
}

template<typename distrNetType>
size_t Controller<distrNetType>::RandomInt(size_t maxValue) { return std::rand() % maxValue; }

template<typename distrNetType>
size_t Controller<distrNetType>::NumberOfFeatures() { return numberOfFeatures; }
