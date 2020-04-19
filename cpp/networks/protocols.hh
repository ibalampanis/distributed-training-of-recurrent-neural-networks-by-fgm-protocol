#ifndef DISTRIBUTED_TRAINING_OF_RECURRENT_NEURAL_NETWORKS_BY_FGM_PROTOCOL_PROTOCOLS_HH
#define DISTRIBUTED_TRAINING_OF_RECURRENT_NEURAL_NETWORKS_BY_FGM_PROTOCOL_PROTOCOLS_HH

#include <cmath>
#include <map>
#include <typeinfo>
#include <typeindex>
#include <stdexcept>
#include <cassert>
#include <ctime>
#include <mlpack/core.hpp>
#include "ddsim/dsarch.hh"
#include "ddsim/dds.hh"
#include "cpp/models/rnn_learner.hh"

namespace protocols {

    using namespace dds;
    using namespace arma;
    using namespace rnn_learner;


    // A channel implementation which accounts a combined network cost.
    // The network cost is computed as follows:
    // for each transmit of b bytes, there is a total charge of header * ceiling(b/MSS) bytes.
    // This cost is resembling the TCP segment cost.
    struct TcpChannel : channel {
        static constexpr size_t tcpHeaderBytes = 40;
        static constexpr size_t tcpMsgSize = 1024;

        TcpChannel(host *src, host *dst, rpcc_t endp);

        void transmit(size_t msg_size) override;

        size_t TcpBytes() const;

    protected:
        size_t tcpBytes;
    };

    struct DoubleValue {

        const double value;

        explicit DoubleValue(double val);

        size_t byte_size() const;
    };

    struct IntValue {

        const size_t value;

        explicit IntValue(size_t val);

        size_t byte_size() const;
    };

    // Wrapper for a state parameters.
    // This class wraps a reference to the parameters of a model together with a count
    // of the updates it contains since the last synchronization.
    struct ModelState {
        const arma::mat _model;
        size_t updates;

        ModelState(arma::mat _mdl, size_t _updates);

        size_t byte_size() const;
    };

    struct MatrixMessage {
        const arma::mat &sub_params;

        explicit MatrixMessage(const arma::mat &sb_prms);

        size_t byte_size() const;
    };

    // The base class of a safezone function for machine learning purposes.
    struct SafezoneFunction {

        arma::mat globalModel;          // The global model.

        explicit SafezoneFunction(arma::mat mdl);

        ~SafezoneFunction();

        arma::mat GlobalModel() const;

        void UpdateDrift(arma::mat &drift, arma::mat &params, float mul);

        virtual float Zeta(const arma::mat &params) { return 0.; }

        virtual size_t RegionAdmissibility(const size_t counter) { return 0; }

        virtual float RegionAdmissibility(const arma::mat &mdl) { return 0.; }

        virtual float RegionAdmissibility(const arma::mat &mdl1, const arma::mat &mdl2) { return 0.; }

        virtual float RegionAdmissibilityReb(const arma::mat &mdl1, const arma::mat &mdl2,
                                             double coef) { return 0.; }

        virtual size_t byte_size() { return 0; }
    };

    // This safezone function implements the algorithm presented in
    // in the paper "Communication-Efficient Distributed Online Prediction
    // by Dynamic Model Synchronization"
    // by Michael Kamp, Mario Boley, Assaf Schuster and Izchak Sharfman.
    struct P2Norm : SafezoneFunction {


        double threshold;               // The threshold of the variance between the models of the network.
        size_t batchSize;               // The number of points seen by the node since the last synchronization.

        // Constructors and Destructor 
        P2Norm(arma::mat GlMd, double thr, size_t batch_sz);

        ~P2Norm();

        float Zeta(const arma::mat &params) override;

        float RegionAdmissibility(const arma::mat &mdl) override;

        float RegionAdmissibility(const arma::mat &mdl1, const arma::mat &mdl2) override;

        float RegionAdmissibilityReb(const arma::mat &mdl1, const arma::mat &mdl2,
                                     double coef) override;

        size_t byte_size() override;
    };

    // A wrapper containing the safezone function for machine learning purposes.
    // It is essentially a wrapper for the more verbose, polymorphic safezone_func API,
    // but it conforms to the standard functional API. It is copyable and in addition, it
    // provides a byte_size() method, making it suitable for integration with the middleware.
    class Safezone {
        SafezoneFunction *szone;        // the safezone function, if any

    public:

        Safezone();

        explicit Safezone(SafezoneFunction *sz);

        ~Safezone();

        // Movable
        Safezone(Safezone &&) noexcept;

        Safezone &operator=(Safezone &&) noexcept;

        // Copyable
        Safezone(const Safezone &);

        Safezone &operator=(const Safezone &);

        void Swap(Safezone &other);

        SafezoneFunction *GetSzone();

        void operator()(arma::mat drift, arma::mat params, float mul);

        size_t operator()(size_t counter);

        float operator()(const arma::mat &mdl);

        float operator()(const arma::mat &mdl1, const arma::mat &mdl2);

        size_t byte_size() const;
    };


    // Base class for a query state object.
    // A query state holds the current global estimate model. It also holds the accuracy of the current global model.
    struct QueryState {

        arma::mat globalModel;      // The global model
        double accuracy;            // The accuracy of the current global model

        // Constructor and Destructor 
        QueryState();

        explicit QueryState(const arma::SizeMat &vsz);

        ~QueryState();

        // Update the global model parameters.
        // After this function, the query estimate, accuracy and safezone should adjust to the new global model.
        void UpdateEstimate(arma::mat mdl);

        // Return a SafezoneFunction object for the safe zone function.
        // The returned object shares state with this object.
        // It is the caller's responsibility to delete the returned object and do so before this object is destroyed.
        SafezoneFunction *Safezone(const string &cfg, string algo);

        size_t byte_size() const;
    };


    // Query and protocol configuration.
    struct ProtocolConfig {
        string cfgfile;             // The JSON file containing the info for the test.
        string networkName;         // The name of the network being queried.
        double precision;
        bool rebalancing;
        string distributedLearningAlgorithm;
    };


    // A base class for the query. Objects inheriting this class must override the virtual methods.
    struct Query {
        // These are attributes requested by the user
        ProtocolConfig config;

        Query(const string &cfg, string nm);

        ~Query();

        static QueryState *CreateQueryState();

        double QueryAccuracy(RnnLearner *rnn, arma::cube &tX, arma::cube &tY);
    };


    // The star network topology using the Geometric Method for Distributed Machine Learning.
    template<typename Net, typename Coord, typename Node>
    struct LearningNetwork : dds::star_network<Net, Coord, Node> {

        typedef Coord coordinator_t;
        typedef Node node_t;
        typedef Net network_t;
        typedef star_network<network_t, coordinator_t, node_t> star_network_t;

        Query *Q;

        // Constructor and Destructor
        LearningNetwork(const set<source_id> &_hids, const string &_name, Query *_Q);

        ~LearningNetwork();

        const ProtocolConfig &Cfg() const;

        channel *CreateChannel(host *src, host *dst, rpcc_t endp) const;

        void StartTraining();

        void WarmupNetwork();

        // This is called to update a specific learning node in the network.
        void TrainNode(size_t node, arma::cube &x, arma::cube &y);

        void ShowTrainingStats();
    };


} // end namespace protocols

#endif