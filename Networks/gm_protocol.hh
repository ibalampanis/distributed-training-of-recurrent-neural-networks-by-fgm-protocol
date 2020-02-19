#ifndef __gm_protocol_HH__
#define __gm_protocol_HH__

#include <cmath>
#include <map>
#include <typeinfo>
#include <typeindex>
#include <stdexcept>
#include <cassert>
#include <ctime>
#include "dsource.hh"
#include "dsarch.hh"
#include "RNN_models/predictor/RNNPredictor.hh"

/**
	\file Distributed stream system architecture simulation classes for distributed deep learning.
  */

namespace gm_protocol {

    using namespace dds;
    using namespace rnn_predictor;
    using std::map;
    using std::string;
    using std::vector;
    using std::cout;
    using std::endl;

/**
	A channel implementation which accounts a combined network cost.

	The network cost is computed as follows:
	for each transmit of b bytes, there is a total charge of
	header * ceiling(b/MSS) bytes.

	This cost is resembling the TPC segment cost.
  */
    struct tcp_channel : channel {
        static constexpr size_t tcp_header_bytes = 40;
        static constexpr size_t tcp_mss = 1024;

        tcp_channel(host *src, host *dst, rpcc_t endp);

        void transmit(size_t msg_size) override;

        inline size_t tcp_bytes() const { return tcp_byts; }

    protected:
        size_t tcp_byts;

    };

    struct float_value {
        const float value;

        inline float_value(float qntm) : value(qntm) {}

        size_t byte_size() const { return sizeof(float); }

    };

    struct increment {
        const int increase;

        inline increment(int inc) : increase(inc) {}

        size_t byte_size() const { return sizeof(int); }
    };

    namespace MlPack_GM_Protocol {

/**
	Wrapper for a state parameters.

	This class wraps a reference to the parameters of a model
	together with a count of the updates it contains since the
	last synchronization.
  */
        struct model_state {
            const vector<arma::mat> &_model;
            size_t updates;

            inline model_state(const vector<arma::mat> &_mdl, size_t _updates)
                    : _model(_mdl), updates(_updates) {}

            size_t byte_size() const;
        };

        struct p_model_state {
            const vector<arma::mat *> &_model;
            size_t updates;

            inline p_model_state(const vector<arma::mat *> &_mdl, size_t _updates)
                    : _model(_mdl), updates(_updates) {}

            size_t byte_size() const;
        };

        struct int_num {
            const size_t number;

            inline int_num(const size_t nb)
                    : number(nb) {}

            size_t byte_size() const;

        };

        struct matrix_message {
            const arma::mat &sub_params;

            inline matrix_message(const arma::mat sb_prms)
                    : sub_params(sb_prms) {}

            size_t byte_size() const;
        };

/**
	The base class of a safezone function for machine learning purposes.
	*/
        struct ml_safezone_function {

            vector<arma::mat> &GlobalModel; // The global model.
            vector<float> hyperparameters; // A vector of hyperparameters.

            ml_safezone_function(vector<arma::mat> &mdl);

            virtual ~ml_safezone_function();

            const vector<arma::mat> &getGlobalModel() const { return GlobalModel; }

            void updateDrift(vector<arma::mat> &drift, vector<arma::mat *> &vars, float mul) const;

            virtual float Zeta(const vector<arma::mat> &pars) const { return 0.; }

            virtual float Zeta(const vector<arma::mat *> &pars) const { return 0.; }

            virtual size_t checkIfAdmissible(const size_t counter) const { return 0.; }

            virtual float checkIfAdmissible(const vector<arma::mat> &mdl) const { return 0.; }

            virtual float checkIfAdmissible(const vector<arma::mat *> &mdl) const { return 0.; }

            virtual float
            checkIfAdmissible(const vector<arma::mat *> &par1, const vector<arma::mat> &par2) const { return 0.; }

            virtual float checkIfAdmissible_reb(const vector<arma::mat *> &par1, const vector<arma::mat> &par2,
                                                float coef) const { return 0.; }

            virtual float checkIfAdmissible_v2(const vector<arma::mat> &drift) const { return 0.; }

            virtual float checkIfAdmissible_v2(const vector<arma::mat *> &drift) const { return 0.; }

            virtual size_t byte_size() const { return 0; }

            vector<float> hyper() const { return hyperparameters; }

            virtual void pr() { cout << endl << "Simple safezone function." << endl; }
        };


/**
	This safezone function just checks the number of points
	a local site has proccesed since the last synchronisation. The threshold
	variable basically indicates the batch size. If the proccesed points reach
	the batch size, then the function returns an inadmissible region.
 */
        struct Batch_Learning : ml_safezone_function {

            size_t threshold; // The maximum number of points fitted by each node before requesting synch from the Hub.

            Batch_Learning(vector<arma::mat> &GlMd);

            Batch_Learning(vector<arma::mat> &GlMd, size_t thr);

            ~Batch_Learning();

            size_t checkIfAdmissible(const size_t counter) const override;

            size_t byte_size() const override;
        };

/**
	This safezone function implements the algorithm presented in
	in the paper "Communication-Efficient Distributed Online Prediction
	by Dynamic Model Synchronization"
	by Michael Kamp, Mario Boley, Assaf Schuster and Izchak Sharfman.
 */
        struct Variance_safezone_func : ml_safezone_function {

            float threshold; // The threshold of the variance between the models of the network.
            size_t batch_size; // The number of points seen by the node since the last synchronization.

            Variance_safezone_func(vector<arma::mat> &GlMd);

            Variance_safezone_func(vector<arma::mat> &GlMd, size_t batch_sz);

            Variance_safezone_func(vector<arma::mat> &GlMd, float thr);

            Variance_safezone_func(vector<arma::mat> &GlMd, float thr, size_t batch_sz);

            ~Variance_safezone_func();

            float Zeta(const vector<arma::mat> &pars) const override;

            float Zeta(const vector<arma::mat *> &pars) const override;

            size_t checkIfAdmissible(const size_t counter) const override;

            float checkIfAdmissible(const vector<arma::mat> &mdl) const override;

            float checkIfAdmissible(const vector<arma::mat *> &mdl) const override;

            float checkIfAdmissible(const vector<arma::mat *> &par1, const vector<arma::mat> &par2) const override;

            float checkIfAdmissible_reb(const vector<arma::mat *> &par1, const vector<arma::mat> &par2,
                                        float coef) const override;

            float checkIfAdmissible_v2(const vector<arma::mat> &drift) const override;

            float checkIfAdmissible_v2(const vector<arma::mat *> &drift) const override;

            size_t byte_size() const override;
        };

/**
	A wrapper containing the safezone function for machine
	learning purposes.

	It is essentially a wrapper for the more verbose, polymorphic \c safezone_func API,
	but it conforms to the standard functional API. It is copyable and in addition, it
	provides a byte_size() method, making it suitable for integration with the middleware.
	*/
        class safezone {
            ml_safezone_function *szone;        // the safezone function, if any
        public:

            /// null state
            safezone();

            ~safezone();

            /// valid safezone
            safezone(ml_safezone_function *sz);
            //~safezone();

            /// Movable
            safezone(safezone &&);

            safezone &operator=(safezone &&);

            /// Copyable
            safezone(const safezone &);

            safezone &operator=(const safezone &);

            inline void swap(safezone &other) {
                std::swap(szone, other.szone);
            }

            ml_safezone_function *getSZone() { return (szone != nullptr) ? szone : nullptr; }

            inline void operator()(vector<arma::mat> &drift, vector<arma::mat *> &vars, float mul) {
                szone->updateDrift(drift, vars, mul);
            }

            inline size_t operator()(const size_t counter) {
                return (szone != nullptr) ? szone->checkIfAdmissible(counter) : NAN;
            }

            inline float operator()(const vector<arma::mat> &mdl) {
                return (szone != nullptr) ? szone->checkIfAdmissible(mdl) : NAN;
            }

            inline float operator()(const vector<arma::mat *> &mdl) {
                return (szone != nullptr) ? szone->checkIfAdmissible(mdl) : NAN;
            }

            inline float operator()(const vector<arma::mat *> &par1, const vector<arma::mat> &par2) {
                return (szone != nullptr) ? szone->checkIfAdmissible(par1, par2) : NAN;
            }

            inline size_t byte_size() const {
                return (szone != nullptr) ? szone->byte_size() : 0;
            }

        };


/**
	Base class for a query state object.

	A query state holds the current global estimate model. It also honls the
	accuracy of the current global model (percentage of correctly classified
	datapoins in case of classification and RMSE score in case of regression).
 */
        struct query_state {
            vector<arma::mat> GlobalModel;  // The global model.
            float accuracy; // The accuracy of the current global model.

            query_state();

            query_state(vector<arma::SizeMat> vsz);

            virtual ~query_state();

            void initializeGlobalModel(vector<arma::SizeMat> vsz);

            /** Update the global model parameters.

                After this function, the query estimate, accuracy and
                safezone should adjust to the new global model. For now
                only the global model is adjusted.
                */
            void update_estimate(vector<arma::mat> &mdl);

            void update_estimate(vector<arma::mat *> &mdl);

            void update_estimate_v2(vector<arma::mat> &mdl);

            void update_estimate_v2(vector<arma::mat *> &mdl);

            /**
                Return a ml_safezone_func for the safe zone function.

                 The returned object shares state with this object.
                 It is the caller's responsibility to delete the returned object,
                and do so before this object is destroyed.
             */
            ml_safezone_function *safezone(string cfg, string algo);

            virtual size_t byte_size() const {
                size_t num_of_params = 0;
                for (arma::mat param:GlobalModel)
                    num_of_params += param.n_elem;
                return (1 + num_of_params) * sizeof(float);
            }

        };


/**
	Query and protocol configuration.
  */
        struct protocol_config {
            string learning_algorithm;              // options : [ PA, KernelPA, MLP, PA_Reg, NN_Reg]
            string distributed_learning_algorithm;  // options : [ Batch_Learning, Variance_Monitoring ]
            string cfgfile;                         // The JSON file containing the info for the test.
            string network_name;                    // The name of the network being queried.

            float precision = 0.01;                 // The precision of the FGM protocol.
            float reb_mult = -1.;                   // The precision of the FGM protocol.
            bool rebalancing = false;               // A boolean determining whether the monitoring protocol should run with rabalancing.
            float beta_mu = 0.5;                    // Beta vector coefficient of rebalancing.
            int max_rebs = 2;                       // Maximum number of rebalances
        };


/**
	A base class for a continuous query.
	Objects inheriting this class must override the virtual methods.
	*/
        struct continuous_query {
            // These are attributes requested by the user
            protocol_config config;

            arma::mat *testSet;         // Test dataset without labels.
            arma::mat *testResponses;   // Labels of the test dataset.

            continuous_query(arma::mat *tSet, arma::mat *tRes, string cfg, string nm);

            virtual ~continuous_query() {}

            void setTestSet(arma::mat *tSet, arma::mat *tRes);

            inline query_state *create_query_state() { return new query_state(); }

            inline query_state *create_query_state(vector<arma::SizeMat> sz) { return new query_state(sz); }

            virtual inline double queryAccuracy(MLPACK_Learner *lnr) { return 0.; }
        };

        struct Classification_query : continuous_query {
            /** Constructor */
            Classification_query(arma::mat *tSet, arma::mat *tRes, string cfg, string nm);

            /** Destructor */
            ~Classification_query() {
                delete testSet;
                delete testResponses;
            }

            double queryAccuracy(MLPACK_Learner *lnr) override;

        };

        struct Regression_query : continuous_query {
            /** Constructor */
            Regression_query(arma::mat *tSet, arma::mat *tRes, string cfg, string nm);

            /** Destructor */
            ~Regression_query() {
                delete testSet;
                delete testResponses;
            }

            double queryAccuracy(MLPACK_Learner *lnr) override;
        };

/**
	The star network topology using the Geometric Method
	for Distributed Machine Learning.

	*/
        template<typename Net, typename Coord, typename Node>
        struct gm_learning_network : star_network<Net, Coord, Node> {
            typedef Coord coordinator_t;
            typedef Node node_t;
            typedef Net network_t;
            typedef star_network <network_t, coordinator_t, node_t> star_network_t;

            continuous_query *Q;

            const protocol_config &cfg() const { return Q->config; }

            gm_learning_network(const set<source_id> &_hids, const string &_name, continuous_query *_Q)
                    : star_network_t(_hids), Q(_Q) {
                this->set_name(_name);
                this->setup(Q);
            }

            channel *create_channel(host *src, host *dst, rpcc_t endp) const override {
                if (!dst->is_mcast())
                    return new tcp_channel(src, dst, endp);
                else
                    return basic_network::create_channel(src, dst, endp);
            }

            // This is called to update a specific learning node in the network.
            void process_record(size_t randSite, arma::mat &batch, arma::mat &labels) {
                this->source_site(this->sites.at(randSite)->site_id())->update_stream(batch, labels);
            }

            virtual void warmup(arma::mat &batch, arma::mat &labels) {
                // let the coordinator initialize the nodes
                this->hub->warmup(batch, labels);
            }

            /// This is called to update a specific learning node in the network.
            void end_warmup() {
                this->hub->end_warmup();
            }

            virtual void start_round() {
                this->hub->start_round();
            }

            virtual void process_fini() {
                this->hub->finish_rounds();
            }

            ~gm_learning_network() { delete Q; }
        };

    } //*  End namespace MlPack_GM_Protocol *//


} // end namespace gm_protocol

#endif