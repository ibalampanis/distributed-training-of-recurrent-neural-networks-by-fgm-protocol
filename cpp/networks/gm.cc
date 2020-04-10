#include <random>
#include "gm.hh"
#include "protocols.cc"

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

void Coordinator::SetupConnections() {

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
            proxy[n].ReceiveGlobalParameters(ModelState(globalLearner->ModelParameters(), 0));
        }
        szSent++;
        proxy[n].Reset(Safezone(safezone));
    }
    numRounds++;
}

void Coordinator::Rebalance(node_t *lvnode) {

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
        for (size_t i = 0; i < Mean.size(); i++)
            Mean.at(i) /= cnt;
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
        numSubrounds++;
    } else {
        // New round
        numViolations = 0;
        query->UpdateEstimate(Mean);
        globalLearner->UpdateModel(query->globalModel);
//        ShowProgress();
        StartRound();
    }

}

void Coordinator::FetchUpdates(node_t *node) {

    ModelState up = proxy[node].SendDrift();
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

    if (SafezoneFunction *entity = (Norm2ndDegree *) safezone) {
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

void Coordinator::ShowProgress() {

    // Query thr accuracy of the global model.
    query->accuracy = Q->QueryAccuracy(globalLearner, testX, testY);

    cout << "\n\t[+]Showing progress statistics of training ..." << endl;
    cout << "\t\t-- Model: Global model" << endl;
    cout << "\t\t-- Network name: " << net()->name() << endl;
    cout << "\t\t-- Accuracy: " << std::setprecision(2) << query->accuracy << "%" << endl;
    cout << "\t\t-- Number of rounds: " << numRounds << endl;
    cout << "\t\t-- Total updates: " << totalUpdates << endl << endl;
}

vector<size_t> Coordinator::UpdateStats() const {
    vector<size_t> stats;
    stats.push_back(numRounds);
    stats.push_back(numSubrounds);
    stats.push_back(szSent);
    stats.push_back(0);
    return stats;
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

//    ShowProgress();
    StartRound();
}

double Coordinator::Accuracy() { return query->accuracy = Q->QueryAccuracy(globalLearner, testX, testY); }

void Coordinator::ShowOverallStats() {

    // Query thr accuracy of the global model.
    query->accuracy = Q->QueryAccuracy(globalLearner, testX, testY);

    // See the total number of points received by all the nodes. For debugging.
    for (auto nd:nodePtr)
        totalUpdates += nd->learner->NumberOfUpdates();

    cout << "\n[+]Overall Training Statistics ..." << endl;
    cout << "\t-- Model: Global model" << endl;
    cout << "\t-- Network name: " << Net()->name() << endl;
    cout << "\t-- Accuracy: " << setprecision(3) << query->accuracy << "%" << endl;
    cout << "\t-- Number of rounds: " << numRounds << endl;
    cout << "\t-- Total updates: " << totalUpdates << endl;

    for (auto nd:nodePtr)
        cout << "\t\t-- Node: " << nd->site_id() << setprecision(2) << " - Usage: "
             << (((double) nd->learner->UsedTimes() / (double) totalUpdates) * 100.) << "%" << endl;

}


/*********************************************
	Learning Node
*********************************************/
LearningNode::LearningNode(LearningNode::network_t *net, source_id hid, LearningNode::query_t *_Q) : local_site(net,
                                                                                                                hid),
                                                                                                     Q(_Q),
                                                                                                     coord(this) {
    coord <<= net->hub;
    InitializeLearner();
    datapointsPassed = 0;
};

const ProtocolConfig &LearningNode::Cfg() const { return Q->config; }

void LearningNode::InitializeLearner() {
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

void LearningNode::UpdateState(arma::cube &x, arma::cube &y) {

    learner->TrainModelByBatch(x, y);
    datapointsPassed += x.n_cols;

    if (SafezoneFunction *funcType = dynamic_cast<Norm2ndDegree *>(szone.GetSzone())) {
        if (szone(datapointsPassed) <= 0) {
            datapointsPassed = 0;
            if (szone(learner->ModelParameters()) < 0.)
                coord.LocalViolation(this);
        }
    } else if (szone(datapointsPassed) <= 0.)
        coord.LocalViolation(this);
}

oneway LearningNode::Reset(const Safezone &newsz) {
    szone = newsz;       // Reset the safezone object
    learner->UpdateModel(szone.GetSzone()->GlobalModel()); // Updates the parameters of the local learner
    datapointsPassed = 0;
}

ModelState LearningNode::SendDrift() {
    szone(drift, learner->ModelParameters(), 1.);
    return ModelState(drift, learner->NumberOfUpdates());
}

void LearningNode::ReceiveDrift(const ModelState &mdl) { learner->UpdateModel(mdl._model); }

oneway LearningNode::ReceiveGlobalParameters(const ModelState &params) { learner->UpdateModel(params._model); }

