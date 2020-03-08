#include <random>
#include "gm_network.hh"

using namespace gm_protocol;

/*********************************************
	Network
*********************************************/

gm_protocol::GmNet::GmNet(const set<source_id> &_hids, const string &_name, ContinuousQuery *_Q)
        : gm_learning_network_t(_hids, _name, _Q) {
    this->set_protocol_name("ML_GM");
}


/*********************************************
	Coordinator Side
*********************************************/

Coordinator::Coordinator(network_t *nw, ContinuousQuery *_Q)
        : process(nw), proxy(this),
          Q(_Q),
          k(0),
          num_violations(0), num_rounds(0), num_subrounds(0),
          sz_sent(0), total_updates(0) {
    InitializeLearner();
    query = Q->create_query_state();
    safezone = query->Safezone(cfg().cfgfile, cfg().distributedLearningAlgorithm);
}

Coordinator::~Coordinator() {
    delete safezone;
    delete query;
}

void Coordinator::StartRound() {
    // Send new safezone.
    for (auto n : net()->sites) {
        if (num_rounds == 0) {
            proxy[n].SetHStaticVariables(p_model_state(global_learner->getHModel(), 0));
        }
        sz_sent++;
        proxy[n].Reset(Safezone(safezone));
    }
    num_rounds++;
}

oneway Coordinator::LocalViolation(sender<node_t> ctx) {

    node_t *n = ctx.value;
    num_violations++;

    // Clear
    B.clear(); // Clear the balanced nodes set.
    for (size_t i = 0; i < Mean.size(); i++) {
        Mean.at(i).zeros();
    }
    cnt = 0;

    if (SafezoneFunction *entity = (VarianceSZFunction *) safezone) {
        num_violations = 0;
        FinishRound();
    } else {
        if (num_violations == k) {
            num_violations = 0;
            FinishRound();
        } else {
            KampRebalance(n);
        }
    }
}

void Coordinator::FetchUpdates(node_t *node) {
    ModelState up = proxy[node].GetDrift();
    if (!arma::approx_equal(arma::mat(arma::size(up._model.at(0)), arma::fill::zeros), up._model.at(0), "absdiff",
                            1e-6)) {
        cnt++;
        for (size_t i = 0; i < up._model.size(); i++) {
            Mean.at(i) += up._model.at(i);
        }
    }
    total_updates += up.updates;
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
        if (safezone->CheckIfAdmissible_v2(Mean) > 0. || B.size() == k)
            break;
        for (size_t i = 0; i < Mean.size(); i++)
            Mean.at(i) *= cnt;
    }

    if (B.size() < k) {
        // Rebalancing
        for (size_t i = 0; i < Mean.size(); i++)
            Mean.at(i) += query->globalModel.at(i);
        for (auto n : B) {
            proxy[n].SetDrift(ModelState(Mean, 0));
        }
        num_subrounds++;
    } else {
        // New round
        num_violations = 0;
        query->UpdateEstimateV2(Mean);
        global_learner->UpdateModel(query->globalModel);
        StartRound();
    }

}

// initialize a new round
void Coordinator::FinishRound() {

    // Collect all data
    for (auto n : nodePtr) {
        FetchUpdates(n);
    }
    for (size_t i = 0; i < Mean.size(); i++)
        Mean.at(i) /= cnt;

    // New round
    query->UpdateEstimateV2(Mean);
    global_learner->UpdateModel(query->globalModel);

    StartRound();

}

MatrixMessage Coordinator::HybridDrift(sender<node_t> ctx, IntNum rows, IntNum cols) {
    node_t *n = ctx.value;

    //Virtual Drift. Updating hidden layer matrix.
    global_learner->handleVD(rows.number);
    arma::mat new_hidden = arma::zeros<arma::mat>(rows.number, global_learner->getHModel().at(0)->n_cols);
    new_hidden += global_learner->getHModel().at(0)->rows(global_learner->getHModel().at(0)->n_rows - rows.number,
                                                          global_learner->getHModel().at(0)->n_rows - 1);
    for (auto st : net()->sites) {
        if (st != n) {
            proxy[st].augmentAlphaMatrix(matrix_message(new_hidden));
        }
    }

    // Real Drift. Updating beta vector.
    global_learner->handleRD(cols.number);
    for (auto st : net()->sites) {
        if (st != n) {
            proxy[st].augmentBetaMatrix(cols);
        }
    }
    Mean.at(1).resize(Mean.at(1).n_rows, Mean.at(1).n_cols + cols.number);
    query->globalModel.at(1) = arma::mat(arma::size(*global_learner->GetModelParameters().at(1)), arma::fill::zeros);
    query->globalModel.at(1) += *global_learner->GetModelParameters().at(1);

    return MatrixMessage(new_hidden);
}

MatrixMessage Coordinator::VirtualDrift(sender<node_t> ctx, IntNum rows) {
    // Updating hidden layer matrix.
    node_t *n = ctx.value;
    global_learner->handleVD(rows.number);
    arma::mat new_hidden = arma::zeros<arma::mat>(rows.number, global_learner->getHModel().at(0)->n_cols);
    new_hidden += global_learner->getHModel().at(0)->rows(global_learner->getHModel().at(0)->n_rows - rows.number,
                                                          global_learner->getHModel().at(0)->n_rows - 1);
    for (auto st : net()->sites) {
        if (st != n) {
            proxy[st].augmentAlphaMatrix(matrix_message(new_hidden));
        }
    }
    return MatrixMessage(new_hidden);
}

oneway Coordinator::RealDrift(sender<node_t> ctx, IntNum cols) {
    // Updating beta vector.
    node_t *n = ctx.value;
    global_learner->handleRD(cols.number);
    for (auto st : net()->sites) {
        if (st != n) {
            proxy[st].augmentBetaMatrix(cols);
        }
    }
    Mean.at(1).resize(Mean.at(1).n_rows, Mean.at(1).n_cols + cols.number);
    query->globalModel.at(1) = arma::mat(arma::size(*global_learner->GetModelParameters().at(1)), arma::fill::zeros);
    query->globalModel.at(1) += *global_learner->GetModelParameters().at(1);
}

void Coordinator::Progress() {
    // Query thr accuracy of the global model.
    query->accuracy = Q->QueryAccuracy(global_learner);


    cout << "Global model of network " << net()->name() << "." << endl;
    cout << "accuracy : " << std::setprecision(6) << query->accuracy << "%" << endl;
    cout << "Number of rounds : " << num_rounds << endl;
    cout << "Number of subrounds : " << num_subrounds << endl;
    cout << "Total updates : " << total_updates << endl;
    cout << endl;
}

double Coordinator::GetAccuracy() {
    query->accuracy = Q->QueryAccuracy(global_learner);
    return query->accuracy;
}

vector<size_t> Coordinator::Statistics() const {
    vector<size_t> stats;
    stats.push_back(num_rounds);
    stats.push_back(num_subrounds);
    stats.push_back(sz_sent);
    stats.push_back(0);
    return stats;
}

void Coordinator::FinishRounds() {

    cout << endl;
    cout << "Global model of network " << net()->name() << "." << endl;
    cout << "tests : " << Q->testSet->n_cols << endl;

    // Query thr accuracy of the global model.
    query->accuracy = Q->QueryAccuracy(global_learner);

    // See the total number of points received by all the nodes. For debugging.
    for (auto nd:nodePtr) {
        total_updates += nd->_learner->GetNumberOfUpdates();
    }

    // Print the results.

    cout << "accuracy : " << std::setprecision(6) << query->accuracy << "%" << endl;

    cout << "Number of rounds : " << num_rounds << endl;
    cout << "Number of subrounds : " << num_subrounds << endl;
    cout << "Total updates : " << total_updates << endl;

}

void Coordinator::SetupConnections() {
    using boost::adaptors::map_values;
    proxy.add_sites(net()->sites);
    for (auto n : net()->sites) {
        nodeIndex[n] = nodePtr.size();
        nodePtr.push_back(n);
    }
    k = nodePtr.size();
}

void Coordinator::InitializeLearner() {

    global_learner = new RNNLearner(cfg().cfgfile, RNN<MeanSquaredError<>, HeInitialization>(0));
}


/*********************************************
	Node Side
*********************************************/

oneway LearningNode::Reset(const Safezone &newsz) {
    szone = newsz;       // Reset the safezone object
    datapoints_seen = 0; // Reset the drift vector
    _learner->UpdateModel(szone.GetSZone()->GetGlobalModel()); // Updates the parameters of the local learner
}

ModelState LearningNode::GetDrift() {
    // Getting the drift vector is done as getting the local statistic
    szone(drift, _learner->GetModelParameters(), 1.);
    return ModelState(drift, _learner->GetNumberOfUpdates());
}

void LearningNode::SetDrift(ModelState mdl) {
    // Update the local learner with the model sent by the coordinator
    _learner->UpdateModel(mdl._model);
}

void LearningNode::UpdateStream(arma::mat &batch, arma::mat &labels) {


    size_t alpha_rows = batch.n_rows - _learner->getHModel().at(0)->n_rows;
    size_t beta_cols = labels.n_rows - _learner->GetModelParameters().at(1)->n_cols;

    if (alpha_rows > 0 && beta_cols > 0) {
        arma::mat newA = arma::zeros<arma::mat>(_learner->getHModel().at(0)->n_rows + alpha_rows,
                                                _learner->getHModel().at(0)->n_cols);

        newA.rows(0, _learner->getHModel().at(0)->n_rows - 1) += *_learner->getHModel().at(0);
        newA.rows(_learner->getHModel().at(0)->n_rows, newA.n_rows - 1) +=
                coord.HybridDrift(this, IntNum(alpha_rows), IntNum(beta_cols)).sub_params;

        *_learner->getHModel().at(0) = newA;
    } else if (alpha_rows > 0) {
        arma::mat newA = arma::zeros<arma::mat>(_learner->getHModel().at(0)->n_rows + alpha_rows,
                                                _learner->getHModel().at(0)->n_cols);

        newA.rows(0, _learner->getHModel().at(0)->n_rows - 1) += *_learner->getHModel().at(0);
        newA.rows(_learner->getHModel().at(0)->n_rows, newA.n_rows - 1) +=
                coord.VirtualDrift(this, IntNum(alpha_rows)).sub_params;

        *_learner->getHModel().at(0) = newA;
    } else if (beta_cols > 0) {
        coord.RealDrift(this, IntNum(beta_cols));
    }


    _learner->fit(batch, labels);
    datapoints_seen += batch.n_cols;

    if (SafezoneFunction *entity = static_cast<VarianceSZFunction *>(szone.GetSZone())) {
        if (szone(datapoints_seen) <= 0) {
            datapoints_seen = 0;
            szone(drift, _learner->GetModelParameters(), 1.);
            if (szone.GetSZone()->CheckIfAdmissible_v2(drift) < 0.)
                coord.LocalViolation(this);
        }
    } else {
        if (szone(datapoints_seen) <= 0.)
            coord.LocalViolation(this);
    }
}

oneway LearningNode::SetHStaticVariables(const PModelState &SHParams) {
    _learner->restoreModel(SHParams._model);
}

oneway LearningNode::augmentAlphaMatrix(MatrixMessage params) {
    arma::mat newA = arma::zeros<arma::mat>(_learner->getHModel().at(0)->n_rows + params.sub_params.n_rows,
                                            _learner->getHModel().at(0)->n_cols);
    newA.rows(0, _learner->getHModel().at(0)->n_rows - 1) += *_learner->getHModel().at(0);
    newA.rows(_learner->getHModel().at(0)->n_rows, newA.n_rows - 1) += params.sub_params;
    *_learner->getHModel().at(0) = newA;
}

oneway LearningNode::augmentBetaMatrix(IntNum cols) {
    _learner->handleRD(cols.number);
}

void LearningNode::InitializeLearner() {

    _learner = new RNNLearner(cfg().cfgfile, RNN<MeanSquaredError<>, HeInitialization>(0));

}

void LearningNode::SetupConnections() {
    num_sites = coord.proc()->k;
}


