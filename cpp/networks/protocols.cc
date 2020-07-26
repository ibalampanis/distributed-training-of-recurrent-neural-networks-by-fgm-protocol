#include <jsoncpp/json/json.h>
#include "protocols.hh"
#include "cpp/models/rnn.hh"
#include "ddsim/dsarch.hh"

using namespace protocols;
using namespace dds;
using namespace arma;
using namespace rnn;

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
SafeFunction::SafeFunction(arma::mat mdl) : globalModel(move(mdl)) {}

SafeFunction::~SafeFunction() = default;

arma::mat SafeFunction::GlobalModel() const { return globalModel; }

void SafeFunction::UpdateDrift(arma::mat &drift, arma::mat &params, float mul) {
    using arma::mat;

    if (globalModel.empty()) {
        if (!params.empty())
            globalModel = mat(size(params), arma::fill::zeros);
        else if (!drift.empty())
            globalModel = mat(size(params), arma::fill::zeros);
        else
            return;
    }

    if (drift.empty()) {
        if (!globalModel.empty())
            drift = mat(size(globalModel), arma::fill::zeros);
        else if (!params.empty())
            drift = mat(size(params), arma::fill::zeros);
        else
            return;
    }

    if (params.empty()) {
        if (!globalModel.empty())
            params = mat(size(globalModel), arma::fill::zeros);
        else if (!drift.empty())
            params = mat(size(drift), arma::fill::zeros);
        else
            return;
    }

    mat dr;
    dr = mul * (params - globalModel);
    drift += dr;
}

/*********************************************
	P2Norm Safezone Function
*********************************************/
P2Norm::P2Norm(arma::mat GlMd, double thr, size_t batch_sz) : SafeFunction(GlMd),
                                                              threshold(thr),
                                                              batchSize(batch_sz) {}

P2Norm::~P2Norm() = default;

float P2Norm::Phi(const arma::mat &drift) {

    float leftTerm, rightTerm;

    leftTerm = (float) ((-1 * threshold * arma::norm(globalModel)) -
                        (arma::dot(drift, (globalModel / arma::norm(globalModel)))));

    rightTerm = (float) ((arma::norm((drift + globalModel))) - ((1 + threshold) * arma::norm(globalModel)));

    return std::max(leftTerm, rightTerm);
}

float P2Norm::Phi(const arma::mat &drift, const arma::mat &est) {

    float leftTerm, rightTerm;

    leftTerm = (float) ((-1 * threshold * arma::norm(est)) -
                        (arma::dot(drift, (est / arma::norm(est)))));

    rightTerm = (float) ((arma::norm((drift + est))) - ((1 + threshold) * arma::norm(est)));

    return std::max(leftTerm, rightTerm);
}

float P2Norm::Norm(const arma::mat &drift) {

    if (globalModel.empty() || drift.empty())
        return -1;

    float norm = arma::norm(drift - globalModel);

    return (float) (norm - threshold);
}

float P2Norm::Norm(const arma::mat &drift, const arma::mat &est) {

    if (drift.empty() || est.empty())
        return -1;

    float norm = arma::norm(drift - est);

    return (float) (norm - threshold);
}

size_t P2Norm::byte_size() { return (1 + globalModel.n_elem) * sizeof(float) + sizeof(size_t); }


/*********************************************
	Safezone
*********************************************/
Safezone::Safezone() : safeFunction(nullptr) {}

Safezone::~Safezone() = default;

// valid safezone
Safezone::Safezone(SafeFunction *sz) : safeFunction(sz) {}

// Movable
Safezone::Safezone(Safezone &&other) noexcept { Swap(other); }

Safezone &Safezone::operator=(Safezone &&other) noexcept {
    Swap(other);
    return *this;
}

// Copyable
Safezone::Safezone(const Safezone &other) { safeFunction = other.safeFunction; }

Safezone &Safezone::operator=(const Safezone &other) {

    if (safeFunction != other.safeFunction)
        safeFunction = other.safeFunction;

    return *this;
}

void Safezone::Swap(Safezone &other) { swap(safeFunction, other.safeFunction); }

SafeFunction *Safezone::GetSafeFunction() { return (safeFunction != nullptr) ? safeFunction : nullptr; }

size_t Safezone::byte_size() const { return (safeFunction != nullptr) ? safeFunction->byte_size() : 0; }


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

void QueryState::UpdateEstimate(arma::mat mdl) { globalModel += mdl; }

SafeFunction *QueryState::Safezone(const string &cfg, string algo) {

    Json::Value root;
    ifstream cfgfl(cfg);
    cfgfl >> root;

    auto func = new P2Norm(globalModel,
                           root["query"].get("threshold", -1).asDouble(),
                           root[algo].get("batch_size", -1).asInt());
    return func;

}

size_t QueryState::byte_size() const { return (1 + globalModel.n_elem) * sizeof(float); }


/*********************************************
	Query
*********************************************/
Query::Query(const string &cfg, string nm) {
    try {
        Json::Value root;
        ifstream cfgfl(cfg);
        cfgfl >> root;

        config.distributedLearningAlgorithm = root["net"].get("distributed_learning_algorithm", "trash").asString();
        config.networkName = root["simulations"].get("net_name", "trash").asString();
        config.precision = root[config.distributedLearningAlgorithm].get("precision", 0.01).asFloat();
        config.cfgfile = cfg;
    } catch (...) {}

}

Query::~Query() = default;

QueryState *Query::CreateQueryState() { return new QueryState(); }

double Query::QueryAccuracy(RnnLearner *rnn, arma::cube &tX, arma::cube &tY) { return rnn->MakePrediction(tX, tY); }


/*********************************************
	Learning Network
*********************************************/
template<typename Net, typename Coord, typename Node>
LearningNetwork<Net, Coord, Node>::LearningNetwork(const set<source_id> &_hids, const string &_name, Query *Q)
        : star_network_t(_hids), Q(Q) {
    this->set_name(_name);
    this->setup(Q);
}

template<typename Net, typename Coord, typename Node>
LearningNetwork<Net, Coord, Node>::~LearningNetwork() { delete Q; }

template<typename Net, typename Coord, typename Node>
const ProtocolConfig &LearningNetwork<Net, Coord, Node>::Cfg() const { return Q->config; }

template<typename Net, typename Coord, typename Node>
void LearningNetwork<Net, Coord, Node>::StartTraining() { this->hub->StartRound(); }

template<typename Net, typename Coord, typename Node>
void LearningNetwork<Net, Coord, Node>::WarmupNetwork() { this->hub->WarmupGlobalLearner(); }

template<typename Net, typename Coord, typename Node>
void LearningNetwork<Net, Coord, Node>::TrainNode(size_t node, arma::cube &x, arma::cube &y) {
    this->source_site(this->sites.at(node)->site_id())->UpdateState(x, y);
}

template<typename Net, typename Coord, typename Node>
void LearningNetwork<Net, Coord, Node>::ShowTrainingStats() { this->hub->ShowOverallStats(); }
