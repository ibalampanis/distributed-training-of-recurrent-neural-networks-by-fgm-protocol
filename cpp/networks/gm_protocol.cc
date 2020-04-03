#include <jsoncpp/json/json.h>
#include "gm_protocol.hh"
#include "cpp/models/rnn_learner.hh"
#include "ddsim/dsarch.hh"

using namespace gm_protocol;
using namespace dds;
using namespace arma;
using namespace rnn_learner;
using std::map;
using std::string;
using std::vector;
using std::cout;
using std::endl;

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
	Float Value
*********************************************/
FloatValue::FloatValue(float qntm) : value(qntm) {}

size_t FloatValue::ByteSize() const { return sizeof(float); }


/*********************************************
	Increment
*********************************************/
Increment::Increment(int inc) : increase(inc) {}

size_t Increment::ByteSize() const { return sizeof(int); }


/*********************************************
	Model State
*********************************************/
ModelState::ModelState(arma::mat _mdl, size_t _updates) : _model(_mdl), updates(_updates) {}

size_t ModelState::byte_size() const { return _model.n_elem * sizeof(float); }


/*********************************************
	Matrix Message
*********************************************/
MatrixMessage::MatrixMessage(const arma::mat &sb_prms) : sub_params(sb_prms) {}

size_t MatrixMessage::ByteSize() const { return sizeof(float) * sub_params.n_elem; }


/*********************************************
	Safezone Function
*********************************************/
SafezoneFunction::SafezoneFunction(arma::mat mdl) : globalModel(mdl) {}

SafezoneFunction::~SafezoneFunction() = default;

const arma::mat SafezoneFunction::GlobalModel() const { return globalModel; }

void SafezoneFunction::UpdateDrift(arma::mat drift, arma::mat params, float mul) const {
    drift.clear();
    for (size_t i = 0; i < globalModel.size(); i++) {
        arma::mat dr;
        dr = mul * (params.at(i) - globalModel.at(i));
        drift += dr; // (+= mallon, gia na doume)
    }
}

vector<float> SafezoneFunction::Hyperparameters() const { return hyperparameters; }

void SafezoneFunction::Print() { cout << endl << "Simple safezone function." << endl; }


/*********************************************
	Variance Safezone Function
*********************************************/
VarianceFunction::VarianceFunction(arma::mat GlMd, float thr, size_t batch_sz) : SafezoneFunction(GlMd),
                                                                                 threshold(thr),
                                                                                 batchSize(batch_sz) {
    hyperparameters.push_back(thr);
    hyperparameters.push_back(batch_sz);
}

VarianceFunction::~VarianceFunction() = default;

float VarianceFunction::Zeta(const arma::mat &params) const {
    float res = 0.;

    arma::mat subtr = globalModel - params;
    res += arma::dot(subtr, subtr);

    return std::sqrt(threshold) - std::sqrt(res);
}

float VarianceFunction::CheckIfAdmissibleReb(const arma::mat &par1, const arma::mat &par2,
                                             double coef) const {
    float res = 0.;
    arma::mat subtr = par1 - par2;
    ////subtr *= 2.;
    subtr *= coef;
    res += arma::dot(subtr, subtr);

    ////return 0.5*(std::sqrt(threshold)-std::sqrt(res));
    return coef * (std::sqrt(threshold) - std::sqrt(res));
}

float VarianceFunction::CheckIfAdmissible(arma::mat mdl) const {
    double var = 0.;

    arma::mat sub = mdl - globalModel;
    var += arma::dot(sub, sub);

    return threshold - var;
}

size_t VarianceFunction::ByteSize() const { return (1 + globalModel.n_elem) * sizeof(float) + sizeof(size_t); }


/*********************************************
	Batch Learning Safezone Function
*********************************************/
BatchLearningFunction::BatchLearningFunction(arma::mat GlMd, size_t thr) : SafezoneFunction(GlMd),
                                                                           threshold(thr) {
    hyperparameters.push_back(thr);
}

BatchLearningFunction::~BatchLearningFunction() = default;

size_t BatchLearningFunction::CheckIfAdmissible(const size_t counter) const {
    size_t sz = threshold - counter;
    return sz;
}

size_t BatchLearningFunction::ByteSize() const { return globalModel.n_elem * sizeof(float) + sizeof(size_t); }


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

size_t Safezone::operator()(size_t counter) { return (szone != nullptr) ? szone->CheckIfAdmissible(counter) : NAN; }

float Safezone::operator()(const arma::mat mdl) { return (szone != nullptr) ? szone->CheckIfAdmissible(mdl) : NAN; }

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

void QueryState::UpdateEstimate(arma::mat mdl) {
    for (size_t i = 0; i < mdl.size(); i++)
        globalModel.at(i) += mdl.at(i);
}

SafezoneFunction *QueryState::Safezone(const string &cfg, string algo) {

    Json::Value root;
    std::ifstream cfgfl(cfg);
    cfgfl >> root;

    string algorithm = root[algo].get("algorithm", "Variance_Monitoring").asString();
    if (algorithm == "Batch_Learning") {
        auto safe_zone = new BatchLearningFunction(globalModel, root[algo].get("batch_size", 32).asInt64());
        return safe_zone;
    } else if (algorithm == "Variance_Monitoring") {
        auto safe_zone = new VarianceFunction(globalModel,
                                              root[algo].get("threshold", 1.).asFloat(),
                                              root[algo].get("batch_size", 32).asInt64());
        return safe_zone;
    } else {
        return nullptr;
    }
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

double Query::QueryAccuracy(RnnLearner *rnn) { return rnn->ModelAccuracy(); }


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
void GmLearningNetwork<Net, Coord, Node>::ProcessRecord(size_t randSite, arma::mat &batch, arma::mat &labels) {
    this->source_site(this->sites.at(randSite)->site_id())->update_stream(batch, labels);
}

template<typename Net, typename Coord, typename Node>
void GmLearningNetwork<Net, Coord, Node>::StartRound() { this->hub->StartRound(); }

template<typename Net, typename Coord, typename Node>
void GmLearningNetwork<Net, Coord, Node>::FinishProcess() { this->hub->FinishRounds(); }
