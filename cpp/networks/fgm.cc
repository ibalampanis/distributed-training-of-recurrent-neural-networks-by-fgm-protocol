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
        : process(nw), proxy(this),
          Q(Q),
          k(0),
          nRounds(0), nSubrounds(0),
          nSzSent(0), nUpdates(0) {
    InitializeGlobalLearner();
    query = Q->CreateQueryState();
    safezone = query->Safezone(Cfg().cfgfile, Cfg().distributedLearningAlgorithm);
}

algorithms::fgm::Coordinator::~Coordinator() {
    delete safezone;
    delete query;
}

algorithms::fgm::Coordinator::network_t *
algorithms::fgm::Coordinator::Net() { return dynamic_cast<network_t *>(host::net()); }

const ProtocolConfig &algorithms::fgm::Coordinator::Cfg() const { return Q->config; }

void algorithms::fgm::Coordinator::InitializeGlobalLearner() {

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

void algorithms::fgm::Coordinator::WarmupGlobalLearner() { globalLearner->TrainModelByBatch(trainX, trainY); }

void algorithms::fgm::Coordinator::SetupConnections() {

    proxy.add_sites(Net()->sites);

    for (auto n : Net()->sites) {
        nodeIndex[n] = nodePtr.size();
        nodeBoolDrift[n] = 0;
        nodePtr.push_back(n);
    }
    k = nodePtr.size();
}

void algorithms::fgm::Coordinator::StartRound() {

    cnt = 0;
    counter = 0;

    // Calculating the new phi, quantum and the minimum acceptable value for phi.
    phi = k * safezone->Zeta(query->globalModel);
    quantum = phi / (2 * k);
    assert(quantum > 0);
    barrier = Cfg().precision * phi;

    // Send new safezone.
    for (auto n : Net()->sites) {
        if (nRounds == 0)
            proxy[n].ReceiveGlobalModel(ModelState(globalLearner->ModelParameters(), 0));

        nSzSent++;
        proxy[n].Reset(Safezone(safezone), DoubleValue(quantum));
        nodeBoolDrift[n] = 0;
    }

    nRounds++;
    nSubrounds++;
}

void algorithms::fgm::Coordinator::FetchUpdates(node_t *node) {
    ModelState up = proxy[node].SendDrift();
    if (!arma::approx_equal(arma::mat(arma::size(up._model), arma::fill::zeros), up._model, "absdiff",
                            1e-6)) {

        if (nodeBoolDrift[node] == 0) {
            nodeBoolDrift[node] = 1;
            cnt++;
        }
        params += up._model;
    }
    nUpdates += up.updates;
}

oneway algorithms::fgm::Coordinator::SendIncrement(IntValue inc) {
    counter += inc.value;
    if (counter > k) {
        phi = 0.;

        // Collect all data
        for (auto n : nodePtr)
            phi += proxy[n].SendZetaValue().value;

        if (phi >= barrier) {
            counter = 0;
            quantum = phi / (2 * k);
            assert(quantum > 0);
            // send the new quantum
            for (auto n : nodePtr)
                proxy[n].ReceiveQuantum(DoubleValue(quantum));
            nSubrounds++;
        } else {
            for (auto n : nodePtr)
                FetchUpdates(n);

            FinishRound();
        }
    }
}

void algorithms::fgm::Coordinator::FinishRound() {

    params *= pow(cnt, -1);

    // New round
    query->UpdateEstimate(params);
    globalLearner->UpdateModel(query->globalModel);

//    ShowProgress();
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

algorithms::fgm::LearningNode::LearningNode(network_t *net, source_id hid, continuous_query_t *_Q)
        : local_site(net, hid), Q(_Q), coord(this) {
    coord <<= net->hub;
    InitializeLearner();
    datapointsPassed = 0;
};

const ProtocolConfig &algorithms::fgm::LearningNode::Cfg() const { return Q->config; }

void algorithms::fgm::LearningNode::InitializeLearner() {

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


void algorithms::fgm::LearningNode::UpdateState(arma::cube &x, arma::cube &y) {

    learner->TrainModelByBatch(x, y);
    datapointsPassed += x.n_cols;

    int currentC = std::floor((zeta - szone(learner->ModelParameters(), eDelta)) / quantum);
    if (currentC - counter > 0) {
        coord.SendIncrement(IntValue(currentC - counter));
        counter = currentC;

    }
}

oneway algorithms::fgm::LearningNode::Reset(const Safezone &newsz, DoubleValue qntm) {

    counter = 0;
    szone = newsz;                                                      // Reset the safezone object
    quantum = 1. * qntm.value;                                 // Reset the quantum
    learner->UpdateModel(szone.GetSzone()->GlobalModel());       // Updates the parameters of the local learner
    zeta = szone.GetSzone()->Zeta(learner->ModelParameters());   // Reset zeta

    // Initializng the helping matrices if they are not yet initialized.
    arma::mat m = arma::mat(arma::size(eDelta), arma::fill::zeros);
    // FIXME: max()
    if (arma::max(arma::max(arma::abs(eDelta - m))) < 1e-8) {
        for (size_t i = 0; i < learner->ModelParameters().size(); i++) {
            arma::mat tmp1 = arma::mat(arma::size(learner->ModelParameters()), arma::fill::zeros);
            arma::mat tmp2 = arma::mat(arma::size(learner->ModelParameters()), arma::fill::zeros);
            deltaVector = tmp1;
            eDelta = tmp2;
        }
    }

    // Reseting the E_Delta vector.
    eDelta = learner->ModelParameters();

}

oneway algorithms::fgm::LearningNode::ReceiveQuantum(DoubleValue qntm) {
    counter = 0;    // Reset counter
    quantum = 1. * qntm.value;  // Update the quantum
    zeta = szone(learner->ModelParameters(), eDelta);  // Update zeta
}

ModelState algorithms::fgm::LearningNode::SendDrift() {
    // Getting the delta vector is done as getting the local statistic.
    deltaVector.clear();
    arma::mat dr = arma::mat(arma::size(learner->ModelParameters()), arma::fill::zeros);
    dr = learner->ModelParameters() - eDelta;
    deltaVector = dr;

    return ModelState(deltaVector, learner->NumberOfUpdates());
}

DoubleValue algorithms::fgm::LearningNode::SendZetaValue() {
    return DoubleValue(szone(learner->ModelParameters(), eDelta));
}

oneway algorithms::fgm::LearningNode::ReceiveGlobalModel(const ModelState &params) {
    learner->UpdateModel(params._model);
}




