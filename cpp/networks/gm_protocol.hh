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
#include "ddsim/dsarch.hh"
#include "ddsim/dds.hh"
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
     */
    struct TcpChannel : channel {
        static constexpr size_t tcpHeaderBytes = 40;
        static constexpr size_t tcpMsgSize = 1024;

        TcpChannel(host *src, host *dst, rpcc_t endp);

        void transmit(size_t msg_size) override;

        size_t TcpBytes() const;

    protected:
        size_t tcpBytes;

    };

    struct FloatValue {

        const float value;

        explicit FloatValue(float qntm);

        size_t ByteSize() const;

    };

    struct Increment {

        const int increase;

        explicit Increment(int inc);

        size_t ByteSize() const;
    };

    /**
     * Wrapper for a state parameters.
     * This class wraps a reference to the parameters of a model
     * together with a count of the updates it contains since the
     * last synchronization.
     */
    struct ModelState {
        const arma::mat _model;
        size_t updates;

        ModelState(arma::mat _mdl, size_t _updates);

        size_t byte_size() const;
    };

    struct MatrixMessage {
        const arma::mat &sub_params;

        explicit MatrixMessage(const arma::mat &sb_prms);

        size_t ByteSize() const;
    };

    /**
     * The base class of a safezone function for machine learning purposes.
     */
    struct SafezoneFunction {

        arma::mat globalModel;          // The global model.
        vector<float> hyperparameters; // A vector of hyperparameters.

        explicit SafezoneFunction(arma::mat mdl);

        ~SafezoneFunction();

        const arma::mat GlobalModel() const;

        void UpdateDrift(arma::mat drift, arma::mat params, float mul) const;

        virtual float Zeta(const arma::mat &params) const { return 0.; }

        virtual size_t CheckIfAdmissible(const size_t counter) const { return 0.; }

        virtual float CheckIfAdmissible(const arma::mat mdl) const { return 0.; }


        virtual float CheckIfAdmissibleReb(const arma::mat &par1, const arma::mat &par2,
                                           double coef) const { return 0.; }

        virtual size_t ByteSize() const { return 0; }

        vector<float> Hyperparameters() const;

        static void Print();
    };

    /**
     * This safezone function implements the algorithm presented in
     * in the paper "Communication-Efficient Distributed Online Prediction
     * by Dynamic Model Synchronization"
     * by Michael Kamp, Mario Boley, Assaf Schuster and Izchak Sharfman.
     */
    struct VarianceFunction : SafezoneFunction {

        double threshold; // The threshold of the variance between the models of the network.
        size_t batchSize; // The number of points seen by the node since the last synchronization.

        /** Constructors and Destructor */
        VarianceFunction(arma::mat GlMd, float thr, size_t batch_sz);

        ~VarianceFunction();

        float Zeta(const arma::mat &params) const override;

        float CheckIfAdmissible(arma::mat mdl) const override;

        float CheckIfAdmissibleReb(const arma::mat &par1, const arma::mat &par2,
                                   double coef) const override;

        size_t ByteSize() const override;
    };

    /**
     * This safezone function just checks the number of points
     * a local site has proccesed since the last synchronisation. The threshold
     * variable basically indicates the batch size. If the proccesed points reach
     * the batch size, then the function returns an inadmissible region.
     */
    struct BatchLearningFunction : SafezoneFunction {

        size_t threshold; // The maximum number of points fitted by each node before requesting synch from the Hub.

        /** Constructors and Destructor */
        BatchLearningFunction(arma::mat GlMd, size_t thr);

        ~BatchLearningFunction();

        size_t CheckIfAdmissible(size_t counter) const override;

        size_t ByteSize() const override;
    };

    /**
     * A wrapper containing the safezone function for machine
     * learning purposes.
     *
     * It is essentially a wrapper for the more verbose, polymorphic \c safezone_func API,
     * but it conforms to the standard functional API. It is copyable and in addition, it
     * provides a byte_size() method, making it suitable for integration with the middleware.
     */
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

        float operator()(const arma::mat mdl);

        size_t byte_size() const;

    };


    /**
     * Base class for a query state object.
     *
     * A query state holds the current global estimate model. It also holds the
     * accuracy of the current global model.
     **/
    struct QueryState {

        arma::mat globalModel;      // The global model
        double accuracy;            // The accuracy of the current global model

        /** Constructor and Destructor */
        QueryState();

        explicit QueryState(const arma::SizeMat &vsz);

        ~QueryState();

        void InitializeGlobalModel(const arma::SizeMat &vsz);

        /**
         * Update the global model parameters.
         *
         * After this function, the query estimate, accuracy and
         * safezone should adjust to the new global model.
         */
        void UpdateEstimate(arma::mat mdl);

        /**
         * Return a SafezoneFunction object for the safe zone function.
         *
         * The returned object shares state with this object.
         * It is the caller's responsibility to delete the returned object
         * and do so before this object is destroyed.
         */
        SafezoneFunction *Safezone(const string &cfg, string algo);

        size_t ByteSize() const;

    };


    /**
     * Query and protocol configuration.
     */
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
     */
    struct Query {
        // These are attributes requested by the user
        ProtocolConfig config;

        Query(const string &cfg, string nm);

        ~Query();

        static QueryState *CreateQueryState();

        static QueryState *CreateQueryState(arma::SizeMat sz);

        double QueryAccuracy(RnnLearner *lnr);
    };


    /**
     * The star network topology using the Geometric Method
     * for Distributed Machine Learning.
     */
    template<typename Net, typename Coord, typename Node>
    struct GmLearningNetwork : dds::star_network<Net, Coord, Node> {
        typedef Coord coordinator_t;
        typedef Node node_t;
        typedef Net network_t;
        typedef star_network<network_t, coordinator_t, node_t> star_network_t;

        Query *Q;

        /**
         * Constructor and Destructor
         */
        GmLearningNetwork(const set<source_id> &_hids, const string &_name, Query *_Q);

        ~GmLearningNetwork();

        const ProtocolConfig &Cfg() const;

        channel *CreateChannel(host *src, host *dst, rpcc_t endp) const;

        /**
         * This is called to update a specific learning node in the network.
         */
        void ProcessRecord(size_t randSite, arma::mat &batch, arma::mat &labels);

        void StartRound();

        void FinishProcess();

    };


} // end namespace gm_protocol

#endif