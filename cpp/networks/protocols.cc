#include <jsoncpp/json/json.h>
#include "protocols.hh"
#include "cpp/models/rnn_learner.hh"
#include "ddsim/dsarch.hh"

using namespace gm_protocol;
using namespace dds;
using namespace arma;
using namespace rnn_learner;

/*********************************************
	TCP Channel
*********************************************/
TcpChannel::TcpChannel(host *src, host *dst, rpcc_t endp) : channel(src, dst, endp), tcpBytes(0) {}

void TcpChannel::transmit(size_t msg_size) {
    // Update parent statistics
    channel::transmit(msg_size);

    // Update tcp byte count
    size_t segno = (msg_size + tcpMsgSize - 1) / tcpMsgSize;
    tcpBytes += msg_size + segno * tcpHeaderBytes;
}

size_t TcpChannel::TcpBytes() const { return tcpBytes; }


/*********************************************
	Double Value
*********************************************/
DoubleValue::DoubleValue(double val) : value(val) {}

size_t DoubleValue::byte_size() const { return sizeof(double); }

/*********************************************
	Int Value
*********************************************/
IntValue::IntValue(size_t val) : value(val) {}

size_t IntValue::byte_size() const { return sizeof(int); }


/*********************************************
	Increment
*********************************************/
Increment::Increment(int inc) : increase(inc) {}

size_t Increment::byte_size() const { return sizeof(int); }


/*********************************************
	Model State
*********************************************/
ModelState::ModelState(arma::mat _mdl, size_t _updates) : _model(_mdl), updates(_updates) {}

size_t ModelState::byte_size() const { return _model.n_elem * sizeof(float); }


/*********************************************
	Matrix Message
*********************************************/
MatrixMessage::MatrixMessage(const arma::mat &sb_prms) : sub_params(sb_prms) {}

size_t MatrixMessage::byte_size() const { return sizeof(float) * sub_params.n_elem; }


/*********************************************
	Safezone Function
*********************************************/
SafezoneFunction::SafezoneFunction(arma::mat mdl) : globalModel(move(mdl)) {}

SafezoneFunction::~SafezoneFunction() = default;

arma::mat SafezoneFunction::GlobalModel() const { return globalModel; }

void SafezoneFunction::UpdateDrift(arma::mat &drift, arma::mat &params, float mul) {
    using arma::mat;

    if (globalModel.empty()) {
        if (!params.empty())
            globalModel = mat(size(params), arma::fill::zeros);
        else if (!drift.empty())
            globalModel = mat(size(params), arma::fill::zeros);
        else
            assert(false);
    }

    if (drift.empty()) {
        if (!globalModel.empty())
            drift = mat(size(globalModel), arma::fill::zeros);
        else if (!params.empty())
            drift = mat(size(params), arma::fill::zeros);
        else
            assert(false);
    }

    if (params.empty()) {
        if (!globalModel.empty())
            params = mat(size(globalModel), arma::fill::zeros);
        else if (!drift.empty())
            params = mat(size(drift), arma::fill::zeros);
        else
            assert(false);
    }


    mat dr;
    dr = mul * (params - globalModel);
    drift += dr;
}

/*********************************************
	Norm2ndDegree Safezone Function
*********************************************/
Norm2ndDegree::Norm2ndDegree(arma::mat GlMd, float thr, size_t batch_sz) : SafezoneFunction(GlMd),
                                                                           threshold(thr),
                                                                           batchSize(batch_sz) {}

Norm2ndDegree::~Norm2ndDegree() = default;

float Norm2ndDegree::Zeta(const arma::mat &params) const {
    float res = 0.;

    arma::mat subtr = globalModel - params;
    res += arma::dot(subtr, subtr);

    return float(sqrt(threshold) - sqrt(res));
}

float Norm2ndDegree::RegionAdmissibilityReb(const arma::mat &par1, const arma::mat &par2,
                                            double coef) const {
    float res = 0.;
    arma::mat subtr = par1 - par2;
    subtr *= coef;
    res += arma::dot(subtr, subtr);

    return float(coef * (sqrt(threshold) - sqrt(res)));
}

float Norm2ndDegree::RegionAdmissibility(arma::mat mdl) const {

    if (globalModel.empty() || mdl.empty())
        return -1;

    double dotProduct = 0.;

    arma::mat sub = mdl - globalModel;
    dotProduct = arma::dot(sub, sub);

    return float(sqrt(threshold) - sqrt(dotProduct));
}

size_t Norm2ndDegree::ByteSize() const { return (1 + globalModel.n_elem) * sizeof(float) + sizeof(size_t); }


/*********************************************
	Batch Learning Safezone Function
*********************************************/
BatchLearning::BatchLearning(arma::mat GlMd, size_t thr) : SafezoneFunction(GlMd), threshold(thr) {}

BatchLearning::~BatchLearning() = default;

size_t BatchLearning::RegionAdmissibility(const size_t counter) const { return threshold - counter; }

size_t BatchLearning::ByteSize() const { return globalModel.n_elem * sizeof(float) + sizeof(size_t); }


/*********************************************
	Safezone
*********************************************/
Safezone::Safezone() : szone(nullptr) {}

Safezone::~Safezone() = default;

// valid safezone
Safezone::Safezone(SafezoneFunction *sz) : szone(sz) {}

// Movable
Safezone::Safezone(Safezone &&other) noexcept { Swap(other); }

Safezone &Safezone::operator=(Safezone &&other) noexcept {
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

void Safezone::Swap(Safezone &other) { std::swap(szone, other.szone); }

SafezoneFunction *Safezone::GetSzone() { return (szone != nullptr) ? szone : nullptr; }

void Safezone::operator()(arma::mat drift, arma::mat params, float mul) { szone->UpdateDrift(drift, params, mul); }

size_t Safezone::operator()(size_t counter) { return (szone != nullptr) ? szone->RegionAdmissibility(counter) : NAN; }

float Safezone::operator()(const arma::mat &mdl) { return (szone != nullptr) ? szone->RegionAdmissibility(mdl) : NAN; }

size_t Safezone::byte_size() const { return (szone != nullptr) ? szone->ByteSize() : 0; }


/*********************************************
	Query State
*********************************************/
QueryState::QueryState() { accuracy = .0; }

QueryState::QueryState(const arma::SizeMat &vsz) {
    globalModel.set_size(vsz);
    globalModel.zeros();
    accuracy = .0;
}

QueryState::~QueryState() = default;

void QueryState::InitializeGlobalModel(const arma::SizeMat &vsz) {
    globalModel.set_size(vsz);
    globalModel.zeros();
}

void QueryState::UpdateEstimate(arma::mat mdl) { globalModel += mdl; }

SafezoneFunction *QueryState::Safezone(const string &cfg, string algo) {

    Json::Value root;
    std::ifstream cfgfl(cfg);
    cfgfl >> root;

    string algorithm = root[algo].get("algorithm", "Variance_Monitoring").asString();

    auto func = new Norm2ndDegree(globalModel,
                                  root[algo].get("threshold", 1.).asFloat(),
                                  root[algo].get("batch_size", 16).asInt64());
    return func;

}

size_t QueryState::ByteSize() const { return (1 + globalModel.n_elem) * sizeof(float); }


/*********************************************
	Query
*********************************************/
Query::Query(const string &cfg, string nm) {
    cout << "\t[+]Initializing the query ...";
    try {
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

        cout << " OK." << endl;
    } catch (...) {
        cout << " ERROR." << endl;
    }

}

Query::~Query() = default;

QueryState *Query::CreateQueryState() { return new QueryState(); }

QueryState *Query::CreateQueryState(arma::SizeMat sz) { return new QueryState(sz); }

double Query::QueryAccuracy(RnnLearner *rnn, arma::cube &tX, arma::cube &tY) { return rnn->MakePrediction(tX, tY); }


/*********************************************
	GM Learning Network
*********************************************/
template<typename Net, typename Coord, typename Node>
GmLearningNetwork<Net, Coord, Node>::GmLearningNetwork(const set<source_id> &_hids, const string &_name, Query *_Q)
        : star_network_t(_hids), Q(_Q) {
    this->set_name(_name);
    this->setup(Q);
}

template<typename Net, typename Coord, typename Node>
GmLearningNetwork<Net, Coord, Node>::~GmLearningNetwork() { delete Q; }

template<typename Net, typename Coord, typename Node>
const ProtocolConfig &GmLearningNetwork<Net, Coord, Node>::Cfg() const { return Q->config; }

template<typename Net, typename Coord, typename Node>
channel *GmLearningNetwork<Net, Coord, Node>::CreateChannel(host *src, host *dst, rpcc_t endp) const {
    if (!dst->is_mcast())
        return new TcpChannel(src, dst, endp);
    else
        return CreateChannel(src, dst, endp);
}

template<typename Net, typename Coord, typename Node>
void GmLearningNetwork<Net, Coord, Node>::StartTraining() { this->hub->StartRound(); }

template<typename Net, typename Coord, typename Node>
void GmLearningNetwork<Net, Coord, Node>::TrainNode(size_t node, arma::cube &x, arma::cube &y) {
    this->source_site(this->sites.at(node)->site_id())->UpdateState(x, y);
}

template<typename Net, typename Coord, typename Node>
void GmLearningNetwork<Net, Coord, Node>::ShowTrainingStats() { this->hub->ShowOverallStats(); }
