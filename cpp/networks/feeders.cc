#include "feeders.hh"

using namespace gm_protocol;
using namespace arma;
using namespace dds;
using namespace feeders;
using std::vector;
using std::string;

/**
    Feeder
**/

template<typename distrNetType>
feeders::Feeder<distrNetType>::Feeder(string cfg) : config_file(std::move(cfg)) {
    Json::Value root;
    std::ifstream cfgfile(config_file); // Parse from JSON file.
    cfgfile >> root;

    long int random_seed = (long int) root["simulations"].get("seed", 0).asInt64();
    if (random_seed >= 0) {
        seed = random_seed;
    } else {
        seed = time(&random_seed);
    }
    std::srand(seed);

    negative_labels = root["simulations"].get("negative_labels", true).asBool();
    if (negative_labels != true && negative_labels != false) {
        cout << endl << "Incorrect negative labels." << endl;
        cout << "Acceptable negative labels parameters : true, false" << endl;
        throw;
    }

    string learning_problem = root["simulations"].get("learning_problem", "NoProblem").asString();
    if (learning_problem != "classification" && learning_problem != "regression") {
        cout << endl << "Incorrect learning problem given." << endl;
        cout << "Acceptable learning problems are : 'classification', 'regression'" << endl;
        throw;
    } else {
        if (learning_problem == "regression")
            negative_labels = false;
    }

    // Get the stream distribution.
    uniform_distr = root[root["simulations"].get("stream_distribution", "No_Distribution").asString()]
            .get("uniform", true).asBool();
    if (!uniform_distr) {
        B_prob = root[root["simulations"].get("stream_distribution", "No_Distribution").asString()]
                .get("B_prob", -1.).asFloat();
        if (B_prob > 1. || B_prob < 0.) {
            cout << "Invalid parameter B_prob. Probabilities must be in the interval [0,1]" << endl;
            throw;
        }
        site_ratio = root[root["simulations"].get("stream_distribution", "No_Distribution").asString()]
                .get("site_ratio", -1.).asFloat();
        if (site_ratio > 1. || site_ratio < 0.) {
            cout << "Invalid parameter site_ratio. Ratios must be in the interval [0,1]" << endl;
            throw;
        }
    }

    // Boolean flag to determine if the experiment will log the differential accuracies.
    log_diff_acc = root["simulations"].get("log_diff_acc", false).asBool();

}

template<typename distrNetType>
void Feeder<distrNetType>::InitializeSimulation() {

    cout << "Initializing the star nets..." << endl << endl << endl;
    Json::Value root;
    std::ifstream cfgfile(config_file); // Parse from JSON file.
    cfgfile >> root;

    source_id count = 0;
    size_t number_of_networks = root["simulations"].get("number_of_networks", 0).asInt();
    for (size_t i = 1; i <= number_of_networks; i++) {

        string net_name = root["simulations"].get("net_name_" + std::to_string(i), "NoNet").asString();
        string learning_algorithm = root["gm_network_" + net_name].get("learning_algorithm",
                                                                       "NoAlgo").asString();

        cout << "Initializing the network " << net_name << " with " << learning_algorithm << " learner."
             << endl;

        auto query = new gm_protocol::continuous_query(config_file, net_name);
        AddQuery(query);
        cout << "Query added." << endl;


        source_id number_of_nodes = (source_id) root["gm_network_" + net_name].get("number_of_local_nodes",
                                                                                   1).asInt64();
        set<source_id> node_ids;
        for (source_id j = 1; j <= number_of_nodes; j++) {
            node_ids.insert(count + j);
        }
        count += number_of_nodes + 1; // We add one because of the coordinator.

        auto net = new GM_Net(node_ids, net_name, _query_container.at(i - 1));
        AddNet(net);
        cout << "Net " << net_name << " initialized." << endl << endl;
    }
    for (auto net:_net_container) {
        // Initializing the differential communication statistics.
        stats.push_back(chan_frame(net));
        vector<vector<size_t>> dif_com;
        vector<size_t> dif_msgs;
        vector<size_t> dif_bts;
        dif_msgs.push_back(0);
        dif_bts.push_back(0);
        dif_com.push_back(dif_msgs);
        dif_com.push_back(dif_bts);
        differential_communication.push_back(dif_com);

        vector<double> dif_acc;
        dif_acc.push_back(0.);
        differential_accuracy.push_back(dif_acc);

        // Initializing the stream distributions of the sites of the network.
        if (!uniform_distr) {
            set<size_t> B;
            set<size_t> B_compl;
            vector<set<size_t>> net_distr;
            for (size_t i = 0; i < net->sites.size(); i++) {
                B_compl.insert(i);
            }
            for (size_t i = 0; i < std::floor(net->sites.size() * site_ratio); i++) {
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
void Feeder<distrNetType>::PrintStarNets() const {
    cout << endl << "Printing the nets." << endl;
    cout << "Number of networks : " << _net_container.size() << endl;
    for (auto net:_net_container) {
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
void Feeder<distrNetType>::GatherDifferentialInfo() {
    // Gathering the info of the communication triggered by the streaming batch.
    for (size_t i = 0; i < _net_container.size(); i++) {
        size_t batch_messages = 0;
        size_t batch_bytes = 0;

        for (auto chnl:stats.at(i)) {
            batch_messages += chnl->messages_received();
            batch_bytes += chnl->bytes_received();
        }

        differential_communication.at(i).at(0)
                .push_back(batch_messages - msgs);

        differential_communication.at(i).at(1)
                .push_back(batch_bytes - bts);

        msgs = batch_messages;
        bts = batch_bytes;

        if (log_diff_acc)
            differential_accuracy.at(i).push_back(_net_container.at(i)->hub->getAccuracy());
    }
}

/**
    Random Feeder
**/

template<typename distrNetType>
RandomFeeder<distrNetType>::RandomFeeder(const string &cfg) :Feeder<distrNetType>(cfg) {
    try {
        Json::Value root;
        std::ifstream cfgfile(this->config_file); // Parse from JSON file.
        cfgfile >> root;

        linearly_seperable = root["tests_Generated_Data"].get("linearly_seperable", true).asBool();
        if (linearly_seperable != true && linearly_seperable != false) {
            cout << endl << "Incorrect parameter linearly_seperable." << endl;
            cout << "The linearly_seperable parameter must be a boolean." << endl;
            throw;
        }

        this->batchSize = root["tests_Generated_Data"].get("batch_size", 1).asInt64();
        this->warmupSize = root["tests_Generated_Data"].get("warmup_size", 500).asInt64();

        number_of_features = root["tests_Generated_Data"].get("number_of_features", 20).asInt64();
        if (number_of_features <= 0) {
            cout << endl << "Incorrect parameter number_of_features" << endl;
            cout << "Acceptable number_of_features parameters are all the positive integers." << endl;
            throw;
        }

        test_size = root["tests_Generated_Data"].get("test_size", 100000).asDouble();
        if (test_size < 0) {
            cout << endl << "Incorrect parameter test_size." << endl;
            cout << "Acceptable test_size parameters are all the positive integers." << endl;
            throw;
        }

        std::srand(this->seed);

        cout << "test_size : " << test_size << endl << endl;

        // Create the test dataset.
        MakeTestDataset();
        targets = 1;

    } catch (...) {
        cout << endl << "Something went wrong in Random_Feeder object construction." << endl;
        throw;
    }
}

template<typename distrNetType>
void RandomFeeder<distrNetType>::GenNewTarget() {
    target = arma::zeros<arma::mat>(number_of_features, 1);
    for (size_t i = 0; i < target.n_elem; i++) {
        if ((double) std::rand() / RAND_MAX <= std::sqrt(1 - std::pow(2, -(1 / number_of_features)))) {
            target(i, 0) = 1.;
        }
    }
    cout << "New Target" << endl;
    targets++;
}

template<typename distrNetType>
void RandomFeeder<distrNetType>::MakeTestDataset() {
    GenNewTarget();
    this->testSet = arma::zeros<arma::mat>(number_of_features, this->test_size);
    this->testResponses = arma::zeros<arma::mat>(1, this->test_size);

    for (size_t i = 0; i < test_size; i++) {
        arma::dvec point = GenPoint();
        this->testSet.col(i) = point;
        if (arma::dot(point, target.unsafe_col(0)) >= 1.) {
            this->testResponses(0, i) = 1.;
        } else {
            if (this->negative_labels) {
                this->testResponses(0, i) = -1.;
            }
        }
    }
}

template<typename distrNetType>
arma::dvec RandomFeeder<distrNetType>::GenPoint() {
    arma::dvec point = arma::zeros<arma::dvec>(number_of_features);
    for (size_t j = 0; j < number_of_features; j++) {
        if ((double) std::rand() / RAND_MAX <= std::sqrt(1 - std::pow(2, -(1 / number_of_features)))) {
            point(j) = 1.;
        }
    }
    return point;
}

template<typename distrNetType>
void RandomFeeder<distrNetType>::TrainNetworks() {
    size_t count = 0; // Count the number of processed elements.

    bool warm = false; // Variable indicating if the networks are warmed up.
    size_t degrees = 0; // Number of warmup datapoints read so far by the networks.

    while (count < numOfPoints) {

        if ((double) std::rand() / RAND_MAX <= 0.0001) {
            MakeTestDataset();
        }

        arma::mat point = arma::zeros<arma::mat>(number_of_features, 1);
        arma::mat label = arma::zeros<arma::mat>(1, 1);
        point.col(0) = GenPoint();

        if (arma::dot(point.unsafe_col(0), target.unsafe_col(0)) >= 1.) {
            label(0, 0) = 1.;
        } else {
            if (this->negative_labels) {
                label(0, 0) = -1.;
            }
        }

        // Update the number of processed elements.
        count += 1;

        if (!warm) { // We warm up the networks.
            degrees += 1;
            for (size_t i = 0; i < this->_net_container.size(); i++) {
                if (this->_net_container.at(i)->Q->config.learning_algorithm == "MLP") {
                    label.transform([](double val) { return (val == -1.) ? 1. : val + 1.; });
                    this->_net_container.at(i)->warmup(point, label);
                    label.transform([](double val) { return (val == 1.) ? -1. : val - 1; });
                } else {
                    this->_net_container.at(i)->warmup(point, label);
                }
            }
            if (degrees == this->warmupSize) {
                warm = true;
                for (size_t i = 0; i < this->_net_container.size(); i++) {
                    this->_net_container.at(i)->start_round(); // Each hub initializes the first round.
                }
            }
            continue;
        }

        Train(point, label);

        if (count % 5000 == 0) {
            cout << "count : " << count << endl;
        }

    }

    for (auto net:this->_net_container) {
        net->process_fini();
    }
    count = 0;
    cout << "Targets : " << targets << endl;
}

template<typename distrNetType>
void RandomFeeder<distrNetType>::Train(arma::mat &point, arma::mat &label) {
    for (auto net:this->_net_container) {
        size_t random_node = std::rand() % (net->sites.size());
        if (net->cfg().learning_algorithm == "MLP") {

            /// Transform the label to the mlpack's Neural Net format.
            /// After training the transformation is reversed.
            label.transform([](double val) { return (val == -1.) ? 1. : val + 1.; });
            net->process_record(random_node, point, label);
            label.transform([](double val) { return (val == 1.) ? -1. : val - 1; });

        } else {
            /// Train on data point.
            net->process_record(random_node, point, label);
        }
    }
}

