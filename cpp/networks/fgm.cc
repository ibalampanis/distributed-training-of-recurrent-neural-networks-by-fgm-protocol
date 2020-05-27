#include <random>
#include "fgm.hh"
#include "protocols.cc"

using namespace protocols;
using namespace algorithms::fgm;

/*********************************************
	Network
*********************************************/
FgmNet::FgmNet(const set<source_id> &_hids, const string &_name, Query *Q)
        : fgmLearningNetwork(_hids, _name, Q) {
    this->set_protocol_name("ML_FGM");
}


/*********************************************
	Coordinator
*********************************************/
algorithms::fgm::Coordinator::Coordinator(network_t *nw, Query *Q)
        : process(nw), proxy(this), Q(Q), nRounds(0),
        nSubrounds(0), nSzSent(0), nUpdates(0) {
    InitializeGlobalLearner();
    queryState = Q->CreateQueryState();
    safeFunction = queryState->Safezone(Cfg().cfgfile, Cfg().distributedLearningAlgorithm);
}

algorithms::fgm::Coordinator::~Coordinator() {
    delete safeFunction;
    delete queryState;
}

algorithms::fgm::Coordinator::network_t *
algorithms::fgm::Coordinator::Net() { return dynamic_cast<network_t *>(host::net()); }

const ProtocolConfig &algorithms::fgm::Coordinator::Cfg() const { return Q->config; }

void algorithms::fgm::Coordinator::InitializeGlobalLearner() {

    cout << "\n\t\t[+]Coordinator's neural net ...";
    try {
        Json::Value root;
        ifstream cfgfile(Cfg().cfgfile);
        cfgfile >> root;
        size_t rho = root["hyperparameters"].get("rho", -1).asInt();
        globalLearner = new RnnLearner(Cfg().cfgfile, RNN<MeanSquaredError<>, HeInitialization>(rho));
        globalLearner->BuildModel();

        cout << " OK." << endl;

    } catch (...) {
        cout << " ERROR." << endl;
    }
}

void algorithms::fgm::Coordinator::WarmupGlobalLearner() {
    globalLearner->TrainModelByBatch(trainX, trainY);
    queryState->globalModel = globalLearner->ModelParameters();
    safeFunction->globalModel = queryState->globalModel;

}

void algorithms::fgm::Coordinator::SetupConnections() {

    proxy.add_sites(Net()->sites);

    for (auto n : Net()->sites) {
        nodeIndex[n] = nodePtr.size();
        nodePtr.push_back(n);
    }
    k = nodePtr.size();
}

void algorithms::fgm::Coordinator::StartRound() {

    // Set global counter to zero.
    counter = 0;

    // Calculate the new psi, quantum and the minimum acceptable value for psi (barrier).
    arma::mat zeroDrift;
    zeroDrift.set_size(size(queryState->globalModel));
    zeroDrift.zeros();
    psi = k * safeFunction->Phi(zeroDrift);
    theta = -1 * (psi / (double) (2 * k));
    assert(theta > 0);
    // barrier: The smallest number the zeta function can reach.
    barrier = Cfg().precision * psi;  // Cfg().precision is the epsilon psi and is usually equal to 0.01.

    // Send new safezone to nodes
    for (auto n : Net()->sites) {
        if (nRounds == 0)
            proxy[n].ReceiveGlobalModel(ModelState(globalLearner->ModelParameters(), 0));

        nSzSent++;
        proxy[n].ResetForNewRound(Safezone(safeFunction), DoubleValue(theta));
    }

    nRounds++;
    nSubrounds++;
}

void algorithms::fgm::Coordinator::FetchUpdates(node_t *node) {

    ModelState up = proxy[node].SendDrift();
    if (!arma::approx_equal(arma::mat(arma::size(up._model), arma::fill::zeros), up._model, "absdiff",
                            1e-6)) {

        if (params.empty())
            params = up._model;
        else
            params += up._model;
    }
    nUpdates += up.updates;
}

oneway algorithms::fgm::Coordinator::ReceiveIncrement(IntValue inc) {

    // Add the increment to the global counter.
    counter += inc.value;

    if (counter > k) {
        // Here we have a global violation!

        // Collect Phi(Xi,E) from all sites.
        psi = 0;
        for (auto n : nodePtr)
            if (!isnan(proxy[n].SendZeta().value))
                psi += proxy[n].SendZeta().value;


        if (psi >= barrier) {
            // A new round will should start.

            // Take drifts from nodes and aggregate these to the global model.
            for (auto n : nodePtr)
                FetchUpdates(n);

            FinishRound();
        } else {
            // A new subround will should start.

            // Reset global counter.
            counter = 0;

            // Recalculate the quantum.
            assert(psi < 0);
            theta = -1 * (psi / (double) (2 * k));
            assert(theta > 0);

            // Send the new quantum to the nodes.
            for (auto n : nodePtr)
                proxy[n].ReceiveQuantum(DoubleValue(theta));

            nSubrounds++;
        }
    }
}

void algorithms::fgm::Coordinator::FinishRound() {

//    ShowProgress();

    // Update global model and estimate and start a new round.
    queryState->UpdateEstimate(params);
    globalLearner->UpdateModel(queryState->globalModel);
    StartRound();
}

void algorithms::fgm::Coordinator::ShowOverallStats() {

    for (auto nd:nodePtr)
        nUpdates += nd->learner->NumberOfUpdates();

    cout << "\n[+]Overall Training Statistics ..." << endl;
    cout << "\t-> Model Statistics:" << endl;
    cout << "\t\t-- Model: Global model" << endl;
    cout << "\t\t-- Network name: " << Net()->name() << endl;
    cout << "\t\t-- Number of rounds: " << nRounds << endl;
    cout << "\t\t-- Number of subrounds: " << nSubrounds << endl;
    cout << "\t\t-- Accuracy: " << setprecision(4) << Accuracy() << "%" << endl;
    cout << "\t\t-- Total updates: " << nUpdates << endl;
}

void algorithms::fgm::Coordinator::ShowProgress() {

    for (auto nd:nodePtr)
        nUpdates += nd->learner->NumberOfUpdates();

    cout << "\n\t[+]Showing progress statistics of training ..." << endl;
    cout << "\t-> Model Statistics:" << endl;
    cout << "\t\t-- Model: Global model" << endl;
    cout << "\t\t-- Network name: " << Net()->name() << endl;
    cout << "\t\t-- Number of rounds: " << nRounds << endl;
    cout << "\t\t-- Number of subrounds: " << nSubrounds << endl;
    cout << "\t\t-- Accuracy: " << setprecision(4) << Accuracy() << "%" << endl;
    cout << "\t\t-- Total updates: " << nUpdates << endl;
}

double algorithms::fgm::Coordinator::Accuracy() { return globalLearner->MakePrediction(testX, testY); }


/*********************************************
	Learning Node
*********************************************/
algorithms::fgm::LearningNode::LearningNode(network_t *net, source_id hid, continuous_query_t *Q) : local_site(net,
                                                                                                               hid),
                                                                                                    Q(Q), coord(this),
                                                                                                    localCounter(0),
                                                                                                    theta(0),
                                                                                                    zeta(0) {
    coord <<= net->hub;
    InitializeLearner();
    datapointsPassed = 0;
};

const ProtocolConfig &algorithms::fgm::LearningNode::Cfg() const { return Q->config; }

void algorithms::fgm::LearningNode::InitializeLearner() {

    cout << "\t\t[+]Node's local neural net ...";
    try {
        Json::Value root;
        ifstream cfgfile(Cfg().cfgfile);
        cfgfile >> root;
        string temp = root["hyperparameters"].get("rho", 0).asString();
        int rho = stoi(temp);
        learner = new RnnLearner(Cfg().cfgfile, RNN<MeanSquaredError<>, HeInitialization>(rho));
        learner->BuildModel();
        cout << " OK." << endl;
    }
    catch (...) {
        cout << " ERROR." << endl;
        throw;
    }
}

void algorithms::fgm::LearningNode::UpdateState(arma::cube &x, arma::cube &y) {

    // Train local model with a mini-batch.
    learner->TrainModelByBatch(x, y);
    datapointsPassed += x.n_cols;

    // Get the fresh trained model.
    arma::mat justTrained = learner->ModelParameters();

    // The drift is the difference of the fresh trained model in respect of current estimate.
    drift = justTrained - currentEstimate;

    // Hold the current Phi(Xi,E) in case of violation.
    phi = szone.GetSafeFunction()->Phi(drift, currentEstimate);

    // Calculate the new counter and take the max of this and the last updated.
    size_t currentC = floor((phi - zeta) / theta);

    size_t maxC = std::max(currentC, localCounter);

    if (maxC != localCounter) {
        // Here we have a local violation! Ship this increment to the coordinator.
        size_t incr = maxC - localCounter;
        coord.ReceiveIncrement(IntValue(incr));
        localCounter = currentC;
    }
}

oneway algorithms::fgm::LearningNode::ResetForNewRound(const Safezone &newsz, DoubleValue qntm) {
    // Now, a new round begins.

    // Set the local counter to zero and get theta (quantum) from coordinator.
    localCounter = 0;
    theta = (float) qntm.value;

    // Get the new safezone and update the current estimate.
    szone = newsz;
    learner->UpdateModel(szone.GetSafeFunction()->GlobalModel());
    currentEstimate = szone.GetSafeFunction()->GlobalModel();

    // Recalculate zeta
    drift.set_size(size(learner->ModelParameters()));
    drift.zeros();
    zeta = szone.GetSafeFunction()->Phi(drift, currentEstimate);
}

oneway algorithms::fgm::LearningNode::ReceiveQuantum(DoubleValue qntm) {
    // Now, a new subround begins.

    // Set the local counter to zero and get theta (quantum) from coordinator.
    localCounter = 0;
    theta = (float) qntm.value;

    // Recalculate zeta
    zeta = szone.GetSafeFunction()->Phi(drift, currentEstimate);
}

ModelState algorithms::fgm::LearningNode::SendDrift() { return ModelState(drift, learner->NumberOfUpdates()); }

DoubleValue algorithms::fgm::LearningNode::SendZeta() { return DoubleValue(phi); }

oneway algorithms::fgm::LearningNode::ReceiveGlobalModel(const ModelState &params) {
    learner->UpdateModel(params._model);
    currentEstimate = params._model;
}
