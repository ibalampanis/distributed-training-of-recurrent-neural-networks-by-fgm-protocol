#include <jsoncpp/json/json.h>
#include "gm_protocol.hh"
#include "cpp/models/rnn_learner.hh"
#include "cpp/networks/dds/dsarch.hh"


using namespace gm_protocol;

/*********************************************
	model_state & p_model_state
*********************************************/

size_t model_state::byte_size() const {
    size_t num_of_params = 0;
    for (arma::mat param:_model) {
        num_of_params += param.n_elem;
    }
    return num_of_params * sizeof(float);
}

size_t p_model_state::byte_size() const {
    size_t num_of_params = 0;
    for (arma::mat *param:_model) {
        num_of_params += param->n_elem;
    }
    return num_of_params * sizeof(float);
}

size_t int_num::byte_size() const {
    return sizeof(size_t);
}

size_t matrix_message::byte_size() const {
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
	Batch Learning
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

size_t BatchLearningSZFunction::byte_size() const {
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
                                                                  batch_size(32) {
    hyperparameters.push_back(1.);
    hyperparameters.push_back(32.);
}

VarianceSZFunction::VarianceSZFunction(vector<arma::mat> &GlMd, size_t batch_sz) : SafezoneFunction(GlMd),
                                                                                   threshold(1.), batch_size(batch_sz) {
    hyperparameters.push_back(1.);
    hyperparameters.push_back(batch_sz);
}

VarianceSZFunction::VarianceSZFunction(vector<arma::mat> &GlMd, float thr) : SafezoneFunction(GlMd), threshold(thr),
                                                                             batch_size(32) {
    hyperparameters.push_back(thr);
    hyperparameters.push_back(32.);
}

VarianceSZFunction::VarianceSZFunction(vector<arma::mat> &GlMd, float thr, size_t batch_sz) : SafezoneFunction(GlMd),
                                                                                              threshold(thr),
                                                                                              batch_size(batch_sz) {
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

size_t VarianceSZFunction::CheckIfAdmissible(const size_t counter) const { return batch_size - counter; }

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

size_t VarianceSZFunction::byte_size() const {
    size_t num_of_params = 0;
    for (arma::mat param:GlobalModel) {
        num_of_params += param.n_elem;
    }
    return (1 + num_of_params) * sizeof(float) + sizeof(size_t);
}

VarianceSZFunction::~VarianceSZFunction() {}


/*********************************************
	safezone
*********************************************/

Safezone::Safezone() : szone(nullptr) {}

Safezone::~Safezone() {}

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
	continuous_query
*********************************************/

continuous_query::continuous_query(string cfg, string nm) {
    cout << "Initializing the query..." << endl;
    Json::Value root;
    std::ifstream cfgfl(cfg);
    cfgfl >> root;

    config.distributed_learning_algorithm = root["gm_network_" + nm].get("distributed_learning_algorithm", "Trash").asString();
    config.network_name = nm;
    config.precision = root[config.distributed_learning_algorithm].get("precision", 0.01).asFloat();
    config.rebalancing = root[config.distributed_learning_algorithm].get("rebalancing", false).asBool();
    config.reb_mult = root[config.distributed_learning_algorithm].get("reb_mult", -1.).asFloat();
    config.beta_mu = root[config.distributed_learning_algorithm].get("beta_mu", 0.5).asFloat();
    config.max_rebs = root[config.distributed_learning_algorithm].get("max_rebs", 2).asInt64();

    config.cfgfile = cfg;

    cout << "Query initialized : ";
    cout << config.distributed_learning_algorithm << ", ";
    cout << config.network_name << ", ";
    cout << config.cfgfile << endl;
}

void continuous_query::setTestSet(arma::mat *tSet, arma::mat *tRes) {
    testSet = tSet;
    testResponses = tRes;
}

double continuous_query::queryAccuracy(RNNLearner *rnn) { return rnn->GetModelAccuracy(); }

/*********************************************
	tcp_channel
*********************************************/

tcp_channel::tcp_channel(host *src, host *dst, rpcc_t endp) : channel(src, dst, endp), tcp_byts(0) {}

void tcp_channel::transmit(size_t msg_size) {
    // update parent statistics
    channel::transmit(msg_size);

    // update tcp byte count
    size_t segno = (msg_size + tcp_mss - 1) / tcp_mss;
    tcp_byts += msg_size + segno * tcp_header_bytes;
}