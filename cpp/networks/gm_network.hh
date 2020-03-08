#ifndef DISTRIBUTED_TRAINING_OF_RECURRENT_NEURAL_NETWORKS_BY_FGM_PROTOCOL_GM_NETWORK_HH
#define DISTRIBUTED_TRAINING_OF_RECURRENT_NEURAL_NETWORKS_BY_FGM_PROTOCOL_GM_NETWORK_HH

#include <boost/range/adaptors.hpp>
#include <boost/shared_ptr.hpp>
#include <random>
#include "gm_protocol.hh"
#include "cpp/models/rnn_learner.hh"

namespace gm_protocol {

    using namespace dds;
    using namespace rnn_learner;
    using std::map;
    using std::cout;
    using std::endl;


    struct Coordinator;
    struct CoordinatorProxy;
    struct LearningNode;
    struct LearningNodeProxy;

    /**
     * This is a GM Network implementation for the classic Geometric Method protocol.
     */
    struct GmNet : GmLearningNetwork<GmNet, Coordinator, LearningNode> {
        typedef ::gm_protocol::GmLearningNetwork<network_t, coordinator_t, node_t> gm_learning_network_t;

        GmNet(const set<source_id> &_hids, const string &_name, ContinuousQuery *_Q);
    };


    /**
     * This is a hub/coordinator implementation for the classic Geometric Method protocol.
     */
    struct Coordinator : process {
        typedef Coordinator coordinator_t;
        typedef LearningNode node_t;
        typedef LearningNodeProxy node_proxy_t;
        typedef GmNet network_t;

        proxy_map<node_proxy_t, node_t> proxy;

        /**
         * Protocol Stuff
         */
        RNNLearner *global_learner;         // ML model
        ContinuousQuery *Q;                 // continuous query
        QueryState *query;                  // current query state
        SafezoneFunction *safezone;         // the safe zone wrapper
        size_t k;                           // number of sites


        map<node_t *, size_t> nodeIndex;    // index the nodes
        vector<node_t *> nodePtr;

        set<node_t *> B;                    // initialized by local_violation(), updated by rebalancing algorithm
        set<node_t *> Bcompl;               // Complement of B, updated by rebalancing algo

        vector<arma::mat> Mean;             // Used to compute the mean model
        size_t num_violations;              // Number of violations in the same round (for rebalancing)

        int cnt;                            // Helping counter.

        /**
         * Statistics
         */
        size_t num_rounds;                  // Total number of rounds
        size_t num_subrounds;               // Total number of subrounds
        size_t sz_sent;                     // Total safe zones sent
        size_t total_updates;               // Number of stream updates received

        Coordinator(network_t *nw, ContinuousQuery *_Q);

        ~Coordinator() override;

        inline network_t *net() { return dynamic_cast<network_t *>(host::net()); }

        inline const ProtocolConfig &cfg() const { return Q->config; }

        /**
         * Initialize learner and  variables.
         */
        void InitializeLearner();

        void SetupConnections();

        /**
          * Initialize a new round.
          */
        void StartRound();

        void FinishRound();

        void FinishRounds();

        /**
         * Rebalance algorithm by Kamp
         */
        void KampRebalance(node_t *n);

        /**
         * Printing and saving the accuracy.
         */
        void Progress();

        /**
         * Getting the accuracy of the global learner.
         */
        double GetAccuracy();

        /**
         * Get the communication statistics of experiment.
         */
        vector<size_t> Statistics() const;

        /**
         * Get a model of a node.
         */
        void FetchUpdates(node_t *n);

        /**
         * Remote call on host violation.
         */
        oneway LocalViolation(sender<node_t> ctx);

        MatrixMessage HybridDrift(sender<node_t> ctx, IntNum rows, IntNum cols);

        MatrixMessage VirtualDrift(sender<node_t> ctx, IntNum rows);

        oneway RealDrift(sender<node_t> ctx, IntNum cols);

    };

    struct CoordinatorProxy : remote_proxy<Coordinator> {
        using coordinator_t = Coordinator;
        REMOTE_METHOD(coordinator_t, LocalViolation);
        REMOTE_METHOD(coordinator_t, HybridDrift);
        REMOTE_METHOD(coordinator_t, VirtualDrift);
        REMOTE_METHOD(coordinator_t, RealDrift);

        CoordinatorProxy(process *c) : remote_proxy<coordinator_t>(c) {}
    };


    /**
     * This is a site/learning node implementation for the classic Geometric Method protocol.
     */
    struct LearningNode : local_site {

        typedef Coordinator coordinator_t;
        typedef LearningNode node_t;
        typedef LearningNodeProxy node_proxy_t;
        typedef GmNet network_t;
        typedef CoordinatorProxy coord_proxy_t;
        typedef ContinuousQuery continuous_query_t;

        continuous_query_t *Q;                // The query management object
        Safezone szone;                     // The safezone object
        RNNLearner *_learner;               // The learning algorithm

        vector<arma::mat> drift;            // The drift of the node

        int num_sites;                      // Number of sites

        size_t datapoints_seen;             // Number of points the node has seen since the last synchronization
        coord_proxy_t coord;                // The proxy of the coordinator/hub

        /**
         * Constructor
         */
        LearningNode(network_t *net, source_id hid, continuous_query_t *_Q)
                : local_site(net, hid), Q(_Q), coord(this) {
            coord <<= net->hub;
            InitializeLearner();
        };

        inline const ProtocolConfig &cfg() const { return Q->config; }

        void InitializeLearner();

        void SetupConnections();

        void UpdateDrift(vector<arma::mat *> &params);

        void UpdateStream(arma::mat &batch, arma::mat &labels);

        /**
         * Remote Methods
         */

        // Call at the start of a round
        oneway Reset(const Safezone &newsz);

        // Transfer data to the coordinator
        ModelState GetDrift();

        // Set the drift vector (for rebalancing)
        void SetDrift(ModelState mdl);

        oneway SetHStaticVariables(const PModelState &SHParams);

    };

    struct LearningNodeProxy : remote_proxy<LearningNode> {
        typedef LearningNode node_t;
        REMOTE_METHOD(node_t, Reset);
        REMOTE_METHOD(node_t, GetDrift);
        REMOTE_METHOD(node_t, SetDrift);
        REMOTE_METHOD(node_t, SetHStaticVariables);

        LearningNodeProxy(process *p) : remote_proxy<node_t>(p) {}
    };


} // gm_protocol

namespace dds {
    template<>
    inline size_t byte_size<gm_protocol::LearningNode *>(
            gm_protocol::LearningNode *const &) { return 4; }
}

#endif