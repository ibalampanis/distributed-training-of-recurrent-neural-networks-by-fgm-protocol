#include <random>
#include "gm_network.hh"
#include "gm_protocol.cc"

using namespace gm_protocol;
using namespace gm_network;

/*********************************************
	Network
*********************************************/
GmNet::GmNet(const set<source_id> &_hids, const string &_name, Query *_Q)
        : gm_learning_network_t(_hids, _name, _Q) {
    this->set_protocol_name("ML_GM");
}


/*********************************************
	Coordinator
*********************************************/
Coordinator::Coordinator(network_t *nw, Query *_Q) : process(nw), proxy(this),
                                                     Q(_Q),
                                                     k(0),
                                                     numViolations(0), numRounds(0), numSubrounds(0),
                                                     szSent(0), totalUpdates(0) {
    InitializeGlobalLearner();
    query = Q->CreateQueryState();
    safezone = query->Safezone(Cfg().cfgfile, Cfg().distributedLearningAlgorithm);
}

Coordinator::~Coordinator() {
    delete safezone;
    delete query;
}

Coordinator::network_t *Coordinator::Net() { return dynamic_cast<network_t *>(host::net()); }

const ProtocolConfig &Coordinator::Cfg() const { return Q->config; }

void Coordinator::InitializeGlobalLearner() {

    Json::Value root;
    std::ifstream cfgfile(Cfg().cfgfile);
    cfgfile >> root;
    string temp = root["hyperparameters"].get("rho", 0).asString();
    int rho = std::stoi(temp);
    globalLearner = new RnnLearner(Cfg().cfgfile, RNN<MeanSquaredError<>, HeInitialization>(rho));
}

void Coordinator::SetupConnections() {

    using boost::adaptors::map_values;

    proxy.add_sites(Net()->sites);
    for (auto n : Net()->sites) {
        nodeIndex[n] = nodePtr.size();
        nodePtr.push_back(n);
    }
    k = nodePtr.size();
}

void Coordinator::StartRound() {
    // Send new safezone.
    for (auto n : Net()->sites) {
        if (numRounds == 0) {
            proxy[n].SetGlobalParameters(ModelState(globalLearner->ModelParameters(), 0));
        }
        szSent++;
        proxy[n].Reset(Safezone(safezone));
    }
    numRounds++;
}

void Coordinator::FinishRounds() {

    cout << endl;
    cout << "Global model of network " << net()->name() << "." << endl;

    // Query thr accuracy of the global model.
    query->accuracy = Q->QueryAccuracy(globalLearner);

    // See the total number of points received by all the nodes. For debugging.
    for (auto nd:nodePtr) {
        totalUpdates += nd->_learner->NumberOfUpdates();
    }

    // Print the results.

    cout << "Accuracy : " << std::setprecision(6) << query->accuracy << "%" << endl;

    cout << "Number of rounds : " << numRounds << endl;
    cout << "Number of subrounds : " << numSubrounds << endl;
    cout << "Total updates : " << totalUpdates << endl;

}

void Coordinator::FinishRound() {

    // Collect all data
    for (auto n : nodePtr) {
        FetchUpdates(n);
    }
    for (size_t i = 0; i < Mean.size(); i++)
        Mean.at(i) /= cnt;

    // New round
    query->UpdateEstimate(Mean);
    globalLearner->UpdateModel(query->globalModel);

    StartRound();

}

void Coordinator::KampRebalance(node_t *lvnode) {

    Bcompl.clear();
    B.insert(lvnode);

    // find a balancing set
    vector<node_t *> nodes;
    nodes.reserve(k);
    for (auto n:nodePtr) {
        if (B.find(n) == B.end())
            nodes.push_back(n);
    }
    assert(nodes.size() == k - 1);

    // permute the order
    std::shuffle(nodes.begin(), nodes.end(), std::mt19937(std::random_device()()));
    assert(nodes.size() == k - 1);
    assert(B.size() == 1);
    assert(Bcompl.empty());

    for (auto n:nodes) {
        Bcompl.insert(n);
    }
    assert(B.size() + Bcompl.size() == k);
    FetchUpdates(lvnode);
    for (auto n:Bcompl) {
        FetchUpdates(n);
        B.insert(n);
        for (size_t i = 0; i < Mean.size(); i++)
            Mean.at(i) /= cnt;
        if (safezone->CheckIfAdmissible(Mean) > 0. || B.size() == k)
            break;
        for (auto &i : Mean)
            i *= cnt;
    }

    if (B.size() < k) {
        // Rebalancing
        for (size_t i = 0; i < Mean.size(); i++)
            Mean.at(i) += query->globalModel.at(i);
        for (auto n : B) {
            proxy[n].SetDrift(ModelState(Mean, 0));
        }
        numSubrounds++;
    } else {
        // New round
        numViolations = 0;
        query->UpdateEstimate(Mean);
        globalLearner->UpdateModel(query->globalModel);
        StartRound();
    }

}

void Coordinator::Progress() {
    // Query thr accuracy of the global model.
    query->accuracy = Q->QueryAccuracy(globalLearner);


    cout << "Global model of network " << net()->name() << "." << endl;
    cout << "Accuracy : " << std::setprecision(2) << query->accuracy << "%" << endl;
    cout << "Number of rounds : " << numRounds << endl;
    cout << "Number of subrounds : " << numSubrounds << endl;
    cout << "Total updates : " << totalUpdates << endl;
    cout << endl;
}

double Coordinator::Accuracy() {
    query->accuracy = Q->QueryAccuracy(globalLearner);
    return query->accuracy;
}

vector<size_t> Coordinator::Statistics() const {
    vector<size_t> stats;
    stats.push_back(numRounds);
    stats.push_back(numSubrounds);
    stats.push_back(szSent);
    stats.push_back(0);
    return stats;
}

void Coordinator::FetchUpdates(node_t *node) {

    ModelState up = proxy[node].GetDrift();
    if (!arma::approx_equal(arma::mat(arma::size(up._model), arma::fill::zeros), up._model, "absdiff",
                            1e-6)) {
        cnt++;
        Mean += up._model;
    }
    totalUpdates += up.updates;
}

oneway Coordinator::LocalViolation(sender<node_t> ctx) {

    node_t *n = ctx.value;
    numViolations++;

    // Clear
    B.clear(); // Clear the balanced nodes set.
    Mean.zeros();
    cnt = 0;

    if (SafezoneFunction *entity = (VarianceSZFunction *) safezone) {
        numViolations = 0;
        FinishRound();
    } else {
        if (numViolations == k) {
            numViolations = 0;
            FinishRound();
        } else {
            KampRebalance(n);
        }
    }
}
// todo implement Coordinator::Drift()
//oneway Coordinator::Drift(sender<node_t> ctx, int cols) {}


/*********************************************
	Learning Node
*********************************************/
LearningNode::LearningNode(LearningNode::network_t *net, source_id hid, LearningNode::query_t *_Q) : local_site(net,
                                                                                                                hid),
                                                                                                     Q(_Q),
                                                                                                     coord(this) {
    coord <<= net->hub;
    InitializeLearner();
};

const ProtocolConfig &LearningNode::Cfg() const { return Q->config; }

void LearningNode::InitializeLearner() {

    Json::Value root;
    std::ifstream cfgfile(Cfg().cfgfile);
    cfgfile >> root;
    string temp = root["hyperparameters"].get("rho", 0).asString();
    int rho = std::stoi(temp);
    _learner = new RnnLearner(Cfg().cfgfile, RNN<MeanSquaredError<>, HeInitialization>(rho));

    cout << "Local site " << this->name() << " initialized its network." << endl;
}

void LearningNode::SetupConnections() { numSites = coord.proc()->k; }

// todo implement LearningNode::UpdateStream() if is needed
void LearningNode::UpdateStream(arma::mat &batch, arma::mat &labels) {}

// todo implement LearningNode::UpdateDrift()
void UpdateDrift(arma::mat &params) {}

oneway LearningNode::Reset(const Safezone &newsz) {
    szone = newsz;       // Reset the safezone object
    datapoints_seen = 0; // Reset the drift vector
    _learner->UpdateModel(szone.GetSzone()->GlobalModel()); // Updates the parameters of the local learner
}

ModelState LearningNode::GetDrift() {
    // Getting the drift vector is done as getting the local statistic
    szone(drift, _learner->ModelParameters(), 1.);
    return ModelState(drift, _learner->NumberOfUpdates());
}

void LearningNode::SetDrift(const ModelState &mdl) {
    // Update the local learner with the model sent by the coordinator
    _learner->UpdateModel(mdl._model);
}

oneway LearningNode::SetGlobalParameters(const ModelState &params) {
    _learner->UpdateModel(params._model);
}

