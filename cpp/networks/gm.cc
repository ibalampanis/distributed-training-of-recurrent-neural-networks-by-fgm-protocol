#include <random>
#include "gm.hh"
#include "protocols.cc"

using namespace protocols;
using namespace gm;

/*********************************************
	Network
*********************************************/
GmNet::GmNet(const set<source_id> &_hids, const string &_name, Query *_Q)
        : gmLearningNetwork(_hids, _name, _Q) {
    this->set_protocol_name("ML_GM");
}


/*********************************************
	Coordinator
*********************************************/
gm::Coordinator::Coordinator(network_t *nw, Query *_Q) : process(nw), proxy(this), Q(_Q), k(0), numViolations(0),
                                                         nRounds(0), nRebalances(0), nSzSent(0), nUpdates(0) {
    InitializeGlobalLearner();
    query = Q->CreateQueryState();
    safezone = query->Safezone(Cfg().cfgfile, Cfg().distributedLearningAlgorithm);
}

gm::Coordinator::~Coordinator() {
    delete safezone;
    delete query;
}

gm::Coordinator::network_t *gm::Coordinator::Net() { return dynamic_cast<network_t *>(host::net()); }

const ProtocolConfig &gm::Coordinator::Cfg() const { return Q->config; }

void gm::Coordinator::InitializeGlobalLearner() {

    cout << "\n\t\t[+]Coordinator's neural net ...";
    try {
        Json::Value root;
        std::ifstream cfgfile(Cfg().cfgfile);
        cfgfile >> root;
        string temp = root["hyperparameters"].get("rho", 0).asString();
        int rho = std::stoi(temp);
        globalLearner = new RnnLearner(Cfg().cfgfile, RNN<MeanSquaredError<>, HeInitialization>(rho));
        globalLearner->BuildModel();

        cout << " OK." << endl;

    } catch (...) {
        cout << " ERROR." << endl;
    }
}

void gm::Coordinator::SetupConnections() {

    proxy.add_sites(Net()->sites);

    for (auto n : Net()->sites) {
        nodeIndex[n] = nodePtr.size();
        nodePtr.push_back(n);
    }

    k = nodePtr.size();
}

void gm::Coordinator::StartRound() {
    // Send new safezone.
    for (auto n : Net()->sites) {
        if (nRounds == 0) {
            proxy[n].ReceiveGlobalParameters(ModelState(globalLearner->ModelParameters(), 0));
        }
        nSzSent++;
        proxy[n].Reset(Safezone(safezone));
    }
    nRounds++;
}

void gm::Coordinator::Rebalance(node_t *lvnode) {

    nRebalances++;

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
    shuffle(nodes.begin(), nodes.end(), mt19937(random_device()()));

    assert(nodes.size() == k - 1);
    assert(B.size() == 1);
    assert(Bcompl.empty());

    for (auto n:nodes)
        Bcompl.insert(n);

    assert(B.size() + Bcompl.size() == k);

    FetchUpdates(lvnode);

    for (auto n:Bcompl) {
        FetchUpdates(n);
        B.insert(n);
        for (double &i : Mean)
            i /= cnt;
        if (safezone->RegionAdmissibility(Mean) > 0. || B.size() == k)
            break;
        for (auto &i : Mean)
            i *= cnt;
    }

    if (B.size() < k) {
        // Rebalancing
        for (size_t i = 0; i < Mean.size(); i++)
            Mean.at(i) += query->globalModel.at(i);
        for (auto n : B) {
            proxy[n].ReceiveDrift(ModelState(Mean, 0));
        }
    } else {
        // New round
        numViolations = 0;
        query->UpdateEstimate(Mean);
        globalLearner->UpdateModel(query->globalModel);
//        ShowProgress();
        StartRound();
    }

}

void gm::Coordinator::FetchUpdates(node_t *node) {

    ModelState up = proxy[node].SendDrift();
    if (!arma::approx_equal(arma::mat(arma::size(up._model), arma::fill::zeros), up._model, "absdiff",
                            1e-6)) {
        cnt++;
        Mean += up._model;
    }
    nUpdates += up.updates;
}

oneway gm::Coordinator::LocalViolation(sender<node_t> ctx) {

    node_t *n = ctx.value;
    numViolations++;

    // Clear
    B.clear(); // Clear the balanced nodes set.
    Mean.zeros();
    cnt = 0;

    if (SafezoneFunction *funcType = (SquaredNorm *) safezone) {
        numViolations = 0;
        FinishRound();
    } else {
        if (numViolations == k) {
            numViolations = 0;
            FinishRound();
        } else
            Rebalance(n);
    }
}

void gm::Coordinator::ShowProgress() {

    // Query thr accuracy of the global model.
    query->accuracy = Q->QueryAccuracy(globalLearner, testX, testY);

    cout << "\n\t[+]Showing progress statistics of training ..." << endl;
    cout << "\t\t-- Model: Global model" << endl;
    cout << "\t\t-- Network name: " << net()->name() << endl;
    cout << "\t\t-- Accuracy: " << std::setprecision(2) << query->accuracy << "%" << endl;
    cout << "\t\t-- Number of rounds: " << nRounds << endl;
    cout << "\t\t-- Number of rebalances: " << nRebalances << endl;
    cout << "\t\t-- Total updates: " << nUpdates << endl << endl;
}

vector<size_t> gm::Coordinator::UpdateStats() const {
    vector<size_t> stats;
    stats.push_back(nRounds);
    stats.push_back(nRebalances);
    stats.push_back(nSzSent);
    stats.push_back(0);
    return stats;
}

void gm::Coordinator::FinishRound() {

    // Collect all data
    for (auto n : nodePtr) {
        FetchUpdates(n);
    }
    for (double &i : Mean)
        i /= cnt;

    // New round
    query->UpdateEstimate(Mean);
    globalLearner->UpdateModel(query->globalModel);

//    ShowProgress();
    StartRound();
}

double gm::Coordinator::Accuracy() { return query->accuracy = Q->QueryAccuracy(globalLearner, testX, testY); }

void gm::Coordinator::ShowOverallStats() {

    // See the total number of points received by all the nodes. For debugging.
    for (auto nd:nodePtr)
        nUpdates += nd->learner->NumberOfUpdates();

    cout << "\n[+]Overall Training Statistics ..." << endl;
    cout << "\t-> Model Statistics:" << endl;
    cout << "\t\t-- Model: Global model" << endl;
    cout << "\t\t-- Network name: " << Net()->name() << endl;
    cout << "\t\t-- Number of rounds: " << nRounds << endl;
    cout << "\t\t-- Number of rebalances: " << nRebalances << endl;
    cout << "\t\t-- Accuracy: " << setprecision(4) << Accuracy() << "%" << endl;
    cout << "\t\t-- Total updates: " << nUpdates << endl;

    for (auto nd:nodePtr)
        cout << "\t\t\t-- Node: " << nd->site_id() << setprecision(4) << " - Usage: "
             << (((double) nd->learner->UsedTimes() / (double) trainPoints) * 100.) << "%" <<
             " (" << nd->learner->UsedTimes() << " of " << trainPoints << ")" << endl;

}


/*********************************************
	Learning Node
*********************************************/
gm::LearningNode::LearningNode(gm::LearningNode::network_t *net, source_id hid, gm::LearningNode::query_t *_Q)
        : local_site(net,
                     hid),
          Q(_Q),
          coord(this) {
    coord <<= net->hub;
    InitializeLearner();
    datapointsPassed = 0;
};

const ProtocolConfig &gm::LearningNode::Cfg() const { return Q->config; }

void gm::LearningNode::InitializeLearner() {
    cout << "\t\t[+]Node's local neural net ...";
    try {
        Json::Value root;
        std::ifstream cfgfile(Cfg().cfgfile);
        cfgfile >> root;
        string temp = root["hyperparameters"].get("rho", 0).asString();
        int rho = std::stoi(temp);
        learner = new RnnLearner(Cfg().cfgfile, RNN<MeanSquaredError<>, HeInitialization>(rho));
        learner->BuildModel();
        cout << " OK." << endl;
    }
    catch (...) {
        cout << " ERROR." << endl;
        throw;
    }
}

void gm::Coordinator::WarmupGlobalLearner() { globalLearner->TrainModelByBatch(trainX, trainY); }

void gm::LearningNode::UpdateState(arma::cube &x, arma::cube &y) {

    learner->TrainModelByBatch(x, y);
    datapointsPassed += x.n_cols;

    if (szone(datapointsPassed) <= 0) {
        datapointsPassed = 0;
        if (szone(learner->ModelParameters()) < 0.)
            coord.LocalViolation(this);
    }
}

oneway gm::LearningNode::Reset(const Safezone &newsz) {
    szone = newsz;       // Reset the safezone object
    learner->UpdateModel(szone.GetSzone()->GlobalModel()); // Updates the parameters of the local learner
    datapointsPassed = 0;
}

ModelState gm::LearningNode::SendDrift() {
    szone(drift, learner->ModelParameters(), 1.);
    return ModelState(drift, learner->NumberOfUpdates());
}

void gm::LearningNode::ReceiveDrift(const ModelState &mdl) { learner->UpdateModel(mdl._model); }

oneway gm::LearningNode::ReceiveGlobalParameters(const ModelState &params) { learner->UpdateModel(params._model); }

