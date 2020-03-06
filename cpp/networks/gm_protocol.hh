#ifndef DISTRIBUTED_TRAINING_OF_RECURRENT_NEURAL_NETWORKS_BY_FGM_PROTOCOL_GM_PROTOCOL_HH
#define DISTRIBUTED_TRAINING_OF_RECURRENT_NEURAL_NETWORKS_BY_FGM_PROTOCOL_GM_PROTOCOL_HH

#include <cmath>
#include <map>
#include <typeinfo>
#include <typeindex>
#include <stdexcept>
#include <cassert>
#include <ctime>
#include <mlpack/core.hpp>
#include "dds/dsarch.hh"
#include "dds/dds.hh"
#include "cpp/models/rnn_learner.hh"

namespace gm_protocol {

    using namespace dds;
    using namespace arma;
    using namespace rnn_learner;
    using std::map;
    using std::string;
    using std::vector;
    using std::cout;
    using std::endl;

    /**
     * A channel implementation which accounts a combined network cost.
     * The network cost is computed as follows:
     * for each transmit of b bytes, there is a total charge of
     * header * ceiling(b/MSS) bytes.
     *
     * This cost is resembling the TCP segment cost.
     * **/
    struct TcpChannel : channel {
        static constexpr size_t tcp_header_bytes = 40;
        static constexpr size_t tcp_mss = 1024;

        TcpChannel(host *src, host *dst, rpcc_t endp);

        void transmit(size_t msg_size) override;

        size_t GetTcp_byts() const { return tcp_byts; }

    protected:
        size_t tcp_byts;

    };

    struct FloatValue {

        const float value;

        FloatValue(float qntm) : value(qntm) {}

        size_t ByteSize() const { return sizeof(float); }

    };

    struct Increment {

        const int increase;

        inline Increment(int inc) : increase(inc) {}

        size_t ByteSize() const { return sizeof(int); }
    };

    /**
     * Wrapper for a state parameters.
     * This class wraps a reference to the parameters of a model
     * together with a count of the updates it contains since the
     * last synchronization.
     * **/
    struct ModelState {
        const vector<arma::mat> &_model;
        size_t updates;

        ModelState(const vector<arma::mat> &_mdl, size_t _updates) : _model(_mdl), updates(_updates) {}

        size_t GetByteSize() const;
    };

    struct PModelState {
        const vector<arma::mat *> &_model;
        size_t updates;

        PModelState(const vector<arma::mat *> &_mdl, size_t _updates) : _model(_mdl), updates(_updates) {}

        size_t GetByteSize() const;
    };

    struct IntNum {

        const size_t number;

        IntNum(const size_t nb) : number(nb) {}

        size_t GetByteSize() const;

    };

    struct MatrixMessage {
        const arma::mat &sub_params;

        MatrixMessage(const arma::mat sb_prms) : sub_params(sb_prms) {}

        size_t GetByteSize() const;
    };

    /**
     * The base class of a safezone function for machine learning purposes.
     * **/
    struct SafezoneFunction {

        vector<arma::mat> &GlobalModel; // The global model.
        vector<float> hyperparameters; // A vector of hyperparameters.

        SafezoneFunction(vector<arma::mat> &mdl);

        ~SafezoneFunction();

        inline const vector<arma::mat> &GetGlobalModel() const { return GlobalModel; }

        void UpdateDrift(vector<arma::mat> &drift, vector<arma::mat *> &vars, float mul) const;

        float Zeta(const vector<arma::mat> &pars) const { return 0.; }

        float Zeta(const vector<arma::mat *> &pars) const { return 0.; }

        size_t CheckIfAdmissible(const size_t counter) const { return 0.; }

        float CheckIfAdmissible(const vector<arma::mat> &mdl) const { return 0.; }

        float CheckIfAdmissible(const vector<arma::mat *> &mdl) const { return 0.; }

        float CheckIfAdmissible(const vector<arma::mat *> &par1, const vector<arma::mat> &par2) const { return 0.; }

        float CheckIfAdmissible_reb(const vector<arma::mat *> &par1, const vector<arma::mat> &par2,
                                    float coef) const { return 0.; }

        float CheckIfAdmissible_v2(const vector<arma::mat> &drift) const { return 0.; }

        float CheckIfAdmissible_v2(const vector<arma::mat *> &drift) const { return 0.; }

        size_t GetByteSize() const { return 0; }

        vector<float> GetHyperparameters() const { return hyperparameters; }

        void Print() { cout << endl << "Simple safezone function." << endl; }
    };


    /**
     * This safezone function just checks the number of points
     * a local site has proccesed since the last synchronisation. The threshold
     * variable basically indicates the batch size. If the proccesed points reach
     * the batch size, then the function returns an inadmissible region.
     * **/
    struct BatchLearningSZFunction : SafezoneFunction {

        size_t threshold; // The maximum number of points fitted by each node before requesting synch from the Hub.

        BatchLearningSZFunction(vector<arma::mat> &GlMd);

        BatchLearningSZFunction(vector<arma::mat> &GlMd, size_t thr);

        ~BatchLearningSZFunction();

        size_t CheckIfAdmissible(const size_t counter) const;

        size_t GetByteSize() const;
    };

    /**
     * This safezone function implements the algorithm presented in
     * in the paper "Communication-Efficient Distributed Online Prediction
     * by Dynamic Model Synchronization"
     * by Michael Kamp, Mario Boley, Assaf Schuster and Izchak Sharfman.
     * **/
    struct VarianceSZFunction : SafezoneFunction {

        float threshold; // The threshold of the variance between the models of the network.
        size_t batchSize; // The number of points seen by the node since the last synchronization.

        VarianceSZFunction(vector<arma::mat> &GlMd);

        VarianceSZFunction(vector<arma::mat> &GlMd, size_t batch_sz);

        VarianceSZFunction(vector<arma::mat> &GlMd, float thr);

        VarianceSZFunction(vector<arma::mat> &GlMd, float thr, size_t batch_sz);

        ~VarianceSZFunction();

        float Zeta(const vector<arma::mat> &pars) const;

        float Zeta(const vector<arma::mat *> &pars) const;

        size_t CheckIfAdmissible(const size_t counter) const;

        float CheckIfAdmissible(const vector<arma::mat> &mdl) const;

        float CheckIfAdmissible(const vector<arma::mat *> &mdl) const;

        float CheckIfAdmissible(const vector<arma::mat *> &par1, const vector<arma::mat> &par2) const;

        float CheckIfAdmissible_reb(const vector<arma::mat *> &par1, const vector<arma::mat> &par2,
                                    float coef) const;

        float CheckIfAdmissible_v2(const vector<arma::mat> &drift) const;

        float CheckIfAdmissible_v2(const vector<arma::mat *> &drift) const;

        size_t GetByteSize() const;
    };

    /**
     * A wrapper containing the safezone function for machine
     * learning purposes.
     *
     * It is essentially a wrapper for the more verbose, polymorphic \c safezone_func API,
     * but it conforms to the standard functional API. It is copyable and in addition, it
     * provides a byte_size() method, making it suitable for integration with the middleware.
     * **/
    class Safezone {
        SafezoneFunction *szone;        // the safezone function, if any

    public:

        Safezone();

        ~Safezone();

        // Valid safezone
        explicit Safezone(SafezoneFunction *sz);

        // Movable
        Safezone(Safezone &&);

        Safezone &operator=(Safezone &&);

        // Copyable
        Safezone(const Safezone &);

        Safezone &operator=(const Safezone &);

        void Swap(Safezone &other) {
            std::swap(szone, other.szone);
        }

        SafezoneFunction *GetSZone() { return (szone != nullptr) ? szone : nullptr; }

        inline void operator()(vector<arma::mat> &drift, vector<arma::mat *> &vars, float mul) {
            szone->UpdateDrift(drift, vars, mul);
        }

        inline size_t operator()(const size_t counter) {
            return (szone != nullptr) ? szone->CheckIfAdmissible(counter) : NAN;
        }

        inline float operator()(const vector<arma::mat> &mdl) {
            return (szone != nullptr) ? szone->CheckIfAdmissible(mdl) : NAN;
        }

        inline float operator()(const vector<arma::mat *> &mdl) {
            return (szone != nullptr) ? szone->CheckIfAdmissible(mdl) : NAN;
        }

        inline float operator()(const vector<arma::mat *> &par1, const vector<arma::mat> &par2) {
            return (szone != nullptr) ? szone->CheckIfAdmissible(par1, par2) : NAN;
        }

        inline size_t GetByteSize() const {
            return (szone != nullptr) ? szone->GetByteSize() : 0;
        }

    };


    /**
     * Base class for a query state object.
     *
     * A query state holds the current global estimate model. It also holds the
     * accuracy of the current global model.
     * **/
    struct QueryState {
        vector<arma::mat> globalModel;  // The global model.

        float accuracy; // The accuracy of the current global model.

        QueryState();

        explicit QueryState(const vector<arma::SizeMat> &vsz);

        ~QueryState();

        void InitializeGlobalModel(const vector<arma::SizeMat> &vsz);

        /** Update the global model parameters.

            After this function, the query estimate, accuracy and
            safezone should adjust to the new global model.
            */
        void UpdateEstimate(vector<arma::mat> &mdl);

        void UpdateEstimate(vector<arma::mat *> &mdl);

        void UpdateEstimateV2(vector<arma::mat> &mdl);

        void UpdateEstimateV2(vector<arma::mat *> &mdl);

        /**
            Return a ml_safezone_func for the safe zone function.

            The returned object shares state with this object.
            It is the caller's responsibility to delete the returned object,
            and do so before this object is destroyed.
        **/
        SafezoneFunction *Safezone(const string &cfg, string algo);

        virtual size_t GetByteSize() const {
            size_t num_of_params = 0;
            for (arma::mat param:globalModel)
                num_of_params += param.n_elem;
            return (1 + num_of_params) * sizeof(float);
        }

    };


    /**
     * Query and protocol configuration.
     * **/
    struct ProtocolConfig {
        string cfgfile;             // The JSON file containing the info for the test.
        string networkName;        // The name of the network being queried.
        bool rebalancing = false;   // A boolean determining whether the monitoring protocol should run with rabalancing.
        float betaMu = 0.5;        // Beta vector coefficient of rebalancing.
        int maxRebs = 2;           // Maximum number of rebalances
        float precision;
        float reb_mult;
        string learningAlgorithm;
        string distributedLearningAlgorithm;
    };


    /**
     * A base class for a continuous query.
     * Objects inheriting this class must override the virtual methods.
     * **/
    struct ContinuousQuery {
        // These are attributes requested by the user
        ProtocolConfig config;

        arma::mat *testSet;         // Test dataset without labels.
        arma::mat *testResponses;   // Labels of the test dataset.

        ContinuousQuery(const string &cfg, string nm);

        virtual ~ContinuousQuery() = default;

        void SetTestSet(arma::mat *tSet, arma::mat *tRes);

        static inline QueryState *create_query_state() { return new QueryState(); }

        static inline QueryState *create_query_state(vector<arma::SizeMat> sz) { return new QueryState(sz); }

        virtual inline double QueryAccuracy(RNNLearner *lnr);
    };

    /**
     * The star network topology using the Geometric Method
     * for Distributed Machine Learning.
     * */
    template<typename Net, typename Coord, typename Node>
    struct GmLearningNetwork : star_network<Net, Coord, Node> {
        typedef Coord coordinator_t;
        typedef Node node_t;
        typedef Net network_t;
        typedef star_network<network_t, coordinator_t, node_t> star_network_t;

        ContinuousQuery *Q;

        const ProtocolConfig &cfg() const { return Q->config; }

        GmLearningNetwork(const set<source_id> &_hids, const string &_name, ContinuousQuery *_Q)
                : star_network_t(_hids), Q(_Q) {
            this->set_name(_name);
            this->setup(Q);
        }

        channel *CreateChannel(host *src, host *dst, rpcc_t endp) const {
            if (!dst->is_mcast())
                return new TcpChannel(src, dst, endp);
            else
                return CreateChannel(src, dst, endp);
        }

        /** This is called to update a specific learning node in the network. **/
        void ProcessRecord(size_t randSite, arma::mat &batch, arma::mat &labels) {
            this->source_site(this->sites.at(randSite)->site_id())->update_stream(batch, labels);
        }

        void Warmup(arma::mat &batch, arma::mat &labels) {
            // let the coordinator initialize the nodes
            this->hub->warmup(batch, labels);
        }

        /** This is called to update a specific learning node in the network. **/
        void EndWarmup() {
            this->hub->end_warmup();
        }

        void StartRound() {
            this->hub->start_round();
        }

        void FinishProcess() {
            this->hub->finish_rounds();
        }

        ~GmLearningNetwork() { delete Q; }
    };


} // end namespace gm_protocol

#endif