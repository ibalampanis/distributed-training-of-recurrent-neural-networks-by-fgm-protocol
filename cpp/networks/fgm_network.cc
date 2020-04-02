#include <random>
#include "fgm_network.hh"
#include "gm_protocol.cc"

using namespace gm_protocol;
using namespace fgm_network;

/*********************************************
	Network
*********************************************/
FgmNet::FgmNet(const set<source_id> &_hids, const string &_name, Query *_Q)
        : gm_learning_network_t(_hids, _name, _Q) {
    this->set_protocol_name("ML_FGM");
}


/*********************************************
	Coordinator
*********************************************/
Coordinator::Coordinator(network_t *nw, Query *_Q)
        : process(nw), proxy(this),
          Q(_Q),
          k(0),
          numRounds(0), numSubrounds(0),
          szSent(0), totalUpdates(0), numRebalances(0) {
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
        nodeBoolDrift[n] = 0;
        nodePtr.push_back(n);
    }
    k = nodePtr.size();
}

// fixme Coordinator::StartRound()
void Coordinator::StartRound() {

    // Resets.
    cnt = 0;
    counter = 0;
    rebalanced = false;

    // Calculating the new phi, quantum and the minimum acceptable value for phi.
    phi = k * safezone->Zeta(query->globalModel);
    quantum = phi / (2 * k);
    assert(quantum > 0);
    barrier = Cfg().precision * phi;

    // Send new safezone.
    for (auto n : Net()->sites) {
        if (numRounds == 0) {
            proxy[n].SetGlobalParameters(ModelState(globalLearner->ModelParameters(), 0));
        }
        szSent++;
        proxy[n].Reset(Safezone(safezone), FloatValue(quantum));
        nodeBoolDrift[n] = 0;
    }

    numRounds++;
    numSubrounds++;
}

// fixme Coordinator::FetchUpdates()
void Coordinator::FetchUpdates(node_t *node) {
    ModelState up = proxy[node].GetDrift();
    if (!arma::approx_equal(arma::mat(arma::size(up._model), arma::fill::zeros), up._model, "absdiff",
                            1e-6)) {
        if (nodeBoolDrift[node] == 0) {
            nodeBoolDrift[node] = 1;
            cnt++;
        }
        if (Q->config.rebalancing) {
            for (size_t i = 0; i < up._model.size(); i++) {
                params.at(i) += up._model.at(i);
            }
        } else {
            for (size_t i = 0; i < up._model.size(); i++) {
                Beta.at(i) += up._model.at(i);
            }
        }
    }
    totalUpdates += up.updates;
}

// todo implement Coordinator::Drift()
//oneway Coordinator::Drift(sender<node_t> ctx, size_t cols){}
// fixme Coordinator::SendIncrement()
oneway Coordinator::SendIncrement(Increment inc) {
    counter += inc.increase;
    if (counter > k) {
        phi = 0.;
//        if(rebalanced){
//            for(size_t i=0;i<Beta.size();i++){
//                ////temp.at(i) = query->GlobalModel.at(i) + (2./cnt)*Beta.at(i);
//                temp.at(i) = query->GlobalModel.at(i) + std::pow(Q->config.beta_mu*cnt,-1)*Beta.at(i);
//            }
//            ////phi += (cnt/2)*safe_zone->Zeta(temp);
//            phi += std::pow(Q->config.beta_mu*cnt,-1)*cnt*safe_zone->Zeta(temp);
//        }

        // Collect all data
        for (auto n : nodePtr) {
            phi += proxy[n].GetZetaValue().value;
        }
        if (phi >= barrier) {
            counter = 0;
            quantum = phi / (2 * k);
            assert(quantum > 0);
            // send the new quantum
            for (auto n : nodePtr) {
                proxy[n].TakeQuantum(FloatValue(quantum));
            }
            numSubrounds++;
        } else {
            for (auto n : nodePtr) {
                FetchUpdates(n);
            }
            if (Q->config.rebalancing) {
                rebalanced = true;
                Rebalance();
            } else {
                FinishRound();
            }
        }
    }
}

// fixme Coordinator::FinishRound()
void Coordinator::FinishRound() {

    for (size_t i = 0; i < Beta.size(); i++)
        Beta.at(i) *= std::pow(cnt, -1);

    // New round
    query->UpdateEstimate(Beta);
    globalLearner->UpdateModel(query->globalModel);

    StartRound();

}

// todo check it again later
void Coordinator::FinishRounds() {

    cout << endl;
    cout << "Global model of network " << net()->name() << "." << endl;

    // Query thr accuracy of the global model.
    query->accuracy = Q->QueryAccuracy(globalLearner);

    // See the total number of points received by all the nodes. For debugging.
    for (auto nd:nodePtr) {
        totalUpdates += nd->_learner->getNumOfUpdates();
    }


    cout << "Accuracy : " << std::setprecision(6) << query->accuracy << endl;
    cout << "Number of rounds : " << numRounds << endl;
    cout << "Number of subrounds : " << numSubrounds << endl;
    cout << "Total updates : " << totalUpdates << endl;

}
// todo implement Rebalance() if is needed
//void Coordinator::Rebalance() {}

void Coordinator::Progress() {
    // Query thr accuracy of the global model.
    query->accuracy = Q->QueryAccuracy(globalLearner);


    cout << "Global model of network " << net()->name() << "." << endl;
    cout << "accuracy : " << std::setprecision(6) << query->accuracy << "%" << endl;
    cout << "Number of rounds : " << numRounds << endl;
    cout << "Number of subrounds : " << numSubrounds << endl;
    cout << "Number of rebalances : " << numRebalances << endl;
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
    stats.push_back(numRebalances);
    stats.push_back(szSent);
    return stats;
}


/*********************************************
	Learning Node
*********************************************/

LearningNode::LearningNode(network_t *net, source_id hid, continuous_query_t *_Q)
        : local_site(net, hid), Q(_Q), coord(this) {
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
//void LearningNode::UpdateStream(arma::mat &batch, arma::mat &labels) {}

// todo implement LearningNode::UpdateDrift()
//void UpdateDrift(arma::mat &params) {}

oneway LearningNode::Reset(const Safezone &newsz, const FloatValue qntm) {
    counter = 0;
    szone = newsz;                                                    // Reset the safezone object
    quantum = 1. * qntm.value;                                          // Reset the quantum
    _learner->UpdateModel(szone.GetSzone()->GlobalModel());       // Updates the parameters of the local learner
    zeta = szone.GetSzone()->Zeta(_learner->ModelParameters());              // Reset zeta
    rebalanced = false;

    // Initializng the helping matrices if they are not yet initialized.
    arma::mat m = arma::mat(arma::size(eDelta), arma::fill::zeros);
    if (arma::max(arma::max(arma::abs(eDelta - m))) < 1e-8) {
        for (size_t i = 0; i < _learner->ModelParameters().size(); i++) {
            arma::mat tmp1 = arma::mat(arma::size(_learner->ModelParameters()), arma::fill::zeros);
            arma::mat tmp2 = arma::mat(arma::size(_learner->ModelParameters()), arma::fill::zeros);
            deltaVector = tmp1;
            eDelta = tmp2;
        }
    }

    // Reseting the E_Delta vector.
    eDelta = _learner->ModelParameters();

}

// fixme LearningNode::TakeQuantum() - NOTE: args in szone()
oneway LearningNode::TakeQuantum(const FloatValue qntm) {
    counter = 0;    // Reset counter
    quantum = 1. * qntm.value;  // Update the quantum
    zeta = szone(_learner->ModelParameters(), eDelta);  // Update zeta
}
// todo implement LearningNode::Rebalance() if needed
//oneway LearningNode::Rebalance(const FloatValue qntm){}

ModelState LearningNode::GetDrift() {
    // Getting the delta vector is done as getting the local statistic.
    deltaVector.clear();
    arma::mat dr = arma::mat(arma::size(_learner->ModelParameters()), arma::fill::zeros);
    dr = _learner->ModelParameters() - eDelta;
    deltaVector = dr;

    return ModelState(deltaVector, _learner->NumberOfUpdates());
}

// fixme LearningNode::GetZetaValue() - NOTE: args in szone()
FloatValue LearningNode::GetZetaValue() {
    if (rebalanced && arma::approx_equal(arma::mat(arma::size(eDelta), arma::fill::zeros), eDelta, "absdiff", 1e-6)) {
        return FloatValue(szone.GetSzone()->CheckIfAdmissibleReb(_learner->ModelParameters(), eDelta, 1.));
    } else {
        return FloatValue(szone(_learner->ModelParameters(), eDelta));
    }
}

oneway LearningNode::SetGlobalParameters(const ModelState &params) {
    _learner->UpdateModel(params._model);
}




