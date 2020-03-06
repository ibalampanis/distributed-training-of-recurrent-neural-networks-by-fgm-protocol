#include <jsoncpp/json/json.h>
#include "gm_protocol.hh"
#include "cpp/models/rnn_learner.hh"
#include "dds/dsarch.hh"


using namespace gm_protocol;

/*********************************************
	Model State & p ModelState (pointer)
*********************************************/

size_t ModelState::GetByteSize() const {
    size_t num_of_params = 0;
    for (arma::mat param:_model) {
        num_of_params += param.n_elem;
    }
    return num_of_params * sizeof(float);
}

size_t PModelState::GetByteSize() const {
    size_t num_of_params = 0;
    for (arma::mat *param:_model) {
        num_of_params += param->n_elem;
    }
    return num_of_params * sizeof(float);
}

size_t IntNum::GetByteSize() const {
    return sizeof(size_t);
}

size_t MatrixMessage::GetByteSize() const {
    return sizeof(float) * sub_params.n_elem;
}

/*********************************************
	Safezone Function
*********************************************/

SafezoneFunction::SafezoneFunction(vector<arma::mat> &mdl) : GlobalModel(mdl) {}

void SafezoneFunction::UpdateDrift(vector<arma::mat> &drift, vector<arma::mat *> &vars, float mul) const {
    drift.clear();
    for (size_t i = 0; i < GlobalModel.size(); i++) {
        arma::mat dr = arma::mat(arma::size(*vars.at(i)), arma::fill::zeros);
        dr = mul * (*vars.at(i) - GlobalModel.at(i));
        drift.push_back(dr);
    }
}

SafezoneFunction::~SafezoneFunction() {}


/*********************************************
	Batch Learning Safezone Function
*********************************************/

BatchLearningSZFunction::BatchLearningSZFunction(vector<arma::mat> &GlMd) : SafezoneFunction(GlMd), threshold(32) {
    hyperparameters.push_back(32.);
}

BatchLearningSZFunction::BatchLearningSZFunction(vector<arma::mat> &GlMd, size_t thr) : SafezoneFunction(GlMd),
                                                                                        threshold(thr) {
    hyperparameters.push_back(thr);
}

size_t BatchLearningSZFunction::CheckIfAdmissible(const size_t counter) const {
    size_t sz = threshold - counter;
    return sz;
}

size_t BatchLearningSZFunction::GetByteSize() const {
    size_t num_of_params = 0;
    for (arma::mat param:GlobalModel) {
        num_of_params += param.n_elem;
    }
    return num_of_params * sizeof(float) + sizeof(size_t);
}

BatchLearningSZFunction::~BatchLearningSZFunction() {}


/*********************************************
	Variance Safezone Function
*********************************************/

VarianceSZFunction::VarianceSZFunction(vector<arma::mat> &GlMd) : SafezoneFunction(GlMd), threshold(1.),
                                                                  batchSize(32) {
    hyperparameters.push_back(1.);
    hyperparameters.push_back(32.);
}

VarianceSZFunction::VarianceSZFunction(vector<arma::mat> &GlMd, size_t batch_sz) : SafezoneFunction(GlMd),
                                                                                   threshold(1.), batchSize(batch_sz) {
    hyperparameters.push_back(1.);
    hyperparameters.push_back(batch_sz);
}

VarianceSZFunction::VarianceSZFunction(vector<arma::mat> &GlMd, float thr) : SafezoneFunction(GlMd), threshold(thr),
                                                                             batchSize(32) {
    hyperparameters.push_back(thr);
    hyperparameters.push_back(32.);
}

VarianceSZFunction::VarianceSZFunction(vector<arma::mat> &GlMd, float thr, size_t batch_sz) : SafezoneFunction(GlMd),
                                                                                              threshold(thr),
                                                                                              batchSize(batch_sz) {
    hyperparameters.push_back(thr);
    hyperparameters.push_back(batch_sz);
}

float VarianceSZFunction::Zeta(const vector<arma::mat> &mdl) const {
    float res = 0.;
    for (size_t i = 0; i < mdl.size(); i++) {
        arma::mat subtr = GlobalModel.at(i) - mdl.at(i);
        res += arma::dot(subtr, subtr);
    }
    return std::sqrt(threshold) - std::sqrt(res);
}

float VarianceSZFunction::Zeta(const vector<arma::mat *> &mdl) const {
    float res = 0.;
    for (size_t i = 0; i < mdl.size(); i++) {
        arma::mat subtr = GlobalModel.at(i) - *mdl.at(i);
        res += arma::dot(subtr, subtr);
    }
    return std::sqrt(threshold) - std::sqrt(res);
}

size_t VarianceSZFunction::CheckIfAdmissible(const size_t counter) const { return batchSize - counter; }

float VarianceSZFunction::CheckIfAdmissible(const vector<arma::mat> &mdl) const {
    float var = 0.;
    for (size_t i = 0; i < mdl.size(); i++) {
        arma::mat sub = mdl.at(i) - GlobalModel.at(i);
        var += arma::dot(sub, sub);
    }
    return threshold - var;
}

float VarianceSZFunction::CheckIfAdmissible(const vector<arma::mat *> &mdl) const {
    float var = 0.;
    for (size_t i = 0; i < mdl.size(); i++) {
        arma::mat sub = *mdl.at(i) - GlobalModel.at(i);
        var += arma::dot(sub, sub);
    }
    return threshold - var;
}

float VarianceSZFunction::CheckIfAdmissible(const vector<arma::mat *> &par1, const vector<arma::mat> &par2) const {
    float res = 0.;
    for (size_t i = 0; i < par1.size(); i++) {
        arma::mat subtr = *par1.at(i) - par2.at(i);
        res += arma::dot(subtr, subtr);
    }
    return std::sqrt(threshold) - std::sqrt(res);
}

float VarianceSZFunction::CheckIfAdmissible_reb(const vector<arma::mat *> &par1, const vector<arma::mat> &par2,
                                                float coef) const {
    float res = 0.;
    for (size_t i = 0; i < par1.size(); i++) {
        arma::mat subtr = *par1.at(i) - par2.at(i);
        ////subtr *= 2.;
        subtr *= coef;
        res += arma::dot(subtr, subtr);
    }
    ////return 0.5*(std::sqrt(threshold)-std::sqrt(res));
    return coef * (std::sqrt(threshold) - std::sqrt(res));
}

float VarianceSZFunction::CheckIfAdmissible_v2(const vector<arma::mat> &drift) const {
    float var = 0.;
    for (size_t i = 0; i < drift.size(); i++) {
        var += arma::dot(drift.at(i), drift.at(i));
    }
    return threshold - var;
}

float VarianceSZFunction::CheckIfAdmissible_v2(const vector<arma::mat *> &drift) const {
    float var = 0.;
    for (size_t i = 0; i < drift.size(); i++) {
        var += arma::dot(*drift.at(i), *drift.at(i));
    }
    return threshold - var;
}

size_t VarianceSZFunction::GetByteSize() const {
    size_t num_of_params = 0;
    for (const arma::mat &param:GlobalModel) {
        num_of_params += param.n_elem;
    }
    return (1 + num_of_params) * sizeof(float) + sizeof(size_t);
}

VarianceSZFunction::~VarianceSZFunction() = default;

/*********************************************
	Safezone
*********************************************/

Safezone::Safezone() : szone(nullptr) {}

Safezone::~Safezone() = default;

// valid safezone
Safezone::Safezone(SafezoneFunction *sz) : szone(sz) {}

// Movable
Safezone::Safezone(Safezone &&other) { Swap(other); }

Safezone &Safezone::operator=(Safezone &&other) {
    Swap(other);
    return *this;
}

// Copyable
Safezone::Safezone(const Safezone &other) { szone = other.szone; }

Safezone &Safezone::operator=(const Safezone &other) {
    if (szone != other.szone) {
        szone = other.szone;
    }
    return *this;
}

/*********************************************
	Query State
*********************************************/

QueryState::QueryState() { accuracy = 0.0; }

QueryState::QueryState(const vector<arma::SizeMat> &vsz) {
    for (auto sz:vsz)
        globalModel.emplace_back(sz, arma::fill::zeros);
    accuracy = 0.0;
}

void QueryState::InitializeGlobalModel(const vector<arma::SizeMat> &vsz) {
    for (auto sz:vsz)
        globalModel.emplace_back(sz, arma::fill::zeros);
}

void QueryState::UpdateEstimate(vector<arma::mat> &mdl) {
    for (size_t i = 0; i < mdl.size(); i++)
        globalModel.at(i) -= globalModel.at(i) - mdl.at(i);
}

void QueryState::UpdateEstimate(vector<arma::mat *> &mdl) {
    for (size_t i = 0; i < mdl.size(); i++)
        globalModel.at(i) -= globalModel.at(i) - *mdl.at(i);
}

void QueryState::UpdateEstimateV2(vector<arma::mat> &mdl) {
    for (size_t i = 0; i < mdl.size(); i++)
        globalModel.at(i) += mdl.at(i);
}

void QueryState::UpdateEstimateV2(vector<arma::mat *> &mdl) {
    for (size_t i = 0; i < mdl.size(); i++)
        globalModel.at(i) += *mdl.at(i);
}

QueryState::~QueryState() = default;

SafezoneFunction *QueryState::Safezone(const string &cfg, string algo) {

    Json::Value root;
    std::ifstream cfgfl(cfg);
    cfgfl >> root;

    string algorithm = root[algo].get("algorithm", "Variance_Monitoring").asString();
    cout << algorithm << endl;
    if (algorithm == "Batch_Learning") {
        auto safe_zone = new BatchLearningSZFunction(globalModel, root[algo].get("batch_size", 32).asInt64());
        return safe_zone;
    } else if (algorithm == "Variance_Monitoring") {
        auto safe_zone = new VarianceSZFunction(globalModel,
                                                root[algo].get("threshold", 1.).asFloat(),
                                                root[algo].get("batch_size", 32).asInt64());
        return safe_zone;
    } else {
        return nullptr;
    }
}

/*********************************************
	Continuous Query
*********************************************/

ContinuousQuery::ContinuousQuery(const string &cfg, string nm) {
    cout << "Initializing the query..." << endl;
    Json::Value root;
    std::ifstream cfgfl(cfg);
    cfgfl >> root;

    config.learningAlgorithm = root["gm_network_" + nm]
            .get("learning_algorithm", "Trash").asString();
    config.distributedLearningAlgorithm = root["gm_network_" + nm]
            .get("distributed_learning_algorithm", "Trash").asString();
    config.networkName = nm;
    config.precision = root[config.distributedLearningAlgorithm].get("precision", 0.01).asFloat();
    config.rebalancing = root[config.distributedLearningAlgorithm].get("rebalancing", false).asBool();
    config.reb_mult = root[config.distributedLearningAlgorithm].get("reb_mult", -1.).asFloat();
    config.betaMu = root[config.distributedLearningAlgorithm].get("beta_mu", 0.5).asFloat();
    config.maxRebs = root[config.distributedLearningAlgorithm].get("max_rebs", 2).asInt64();

    config.cfgfile = cfg;

    cout << "Query initialized : ";
    cout << config.networkName << ", ";
    cout << config.cfgfile << endl;
}

void ContinuousQuery::SetTestSet(arma::mat *tSet, arma::mat *tRes) {
    testSet = tSet;
    testResponses = tRes;
}

double ContinuousQuery::QueryAccuracy(RNNLearner *rnn) { return rnn->GetModelAccuracy(); }

/*********************************************
	TCP Channel
*********************************************/

TcpChannel::TcpChannel(host *src, host *dst, rpcc_t endp) : channel(src, dst, endp), tcp_byts(0) {}

void TcpChannel::transmit(size_t msg_size) {
    // update parent statistics
    channel::transmit(msg_size);

    // update tcp byte count
    size_t segno = (msg_size + tcp_mss - 1) / tcp_mss;
    tcp_byts += msg_size + segno * tcp_header_bytes;
}