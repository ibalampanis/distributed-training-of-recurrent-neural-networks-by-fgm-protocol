#ifndef DISTRIBUTED_TRAINING_OF_RECURRENT_NEURAL_NETWORKS_BY_FGM_PROTOCOL_GM_NETWORK_HH
#define DISTRIBUTED_TRAINING_OF_RECURRENT_NEURAL_NETWORKS_BY_FGM_PROTOCOL_GM_NETWORK_HH

#include <boost/range/adaptors.hpp>
#include <boost/shared_ptr.hpp>
#include <random>
#include "gm_protocol.hh"
#include "cpp/models/rnn_learner.hh"

namespace gm_network {

    using namespace dds;
    using namespace rnn_learner;
    using namespace gm_protocol;
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
    struct GmNet : gm_protocol::GmLearningNetwork<GmNet, Coordinator, LearningNode> {

        typedef gm_protocol::GmLearningNetwork<network_t, coordinator_t, node_t> gm_learning_network_t;

        GmNet(const set<source_id> &_hids, const string &_name, Query *_Q);
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

        /** Protocol Stuff */
        RNNLearner *globalLearner;         // ML model
        Query *Q;                           // query
        QueryState *query;                  // current query state
        SafezoneFunction *safezone;         // the safe zone wrapper
        size_t k;                           // number of sites


        map<node_t *, size_t> nodeIndex;    // index the nodes
        vector<node_t *> nodePtr;

        set<node_t *> B;                    // initialized by local_violation(), updated by rebalancing algorithm
        set<node_t *> Bcompl;               // Complement of B, updated by rebalancing algo

        arma::mat Mean;                     // Used to compute the mean model
        size_t numViolations;              // Number of violations in the same round (for rebalancing)

        int cnt;                            // Helping counter.

        /** Statistics */
        size_t numRounds;                  // Total number of rounds
        size_t numSubrounds;               // Total number of subrounds
        size_t szSent;                     // Total safe zones sent
        size_t totalUpdates;               // Number of stream updates received

        /** Constructor and Destructor */
        Coordinator(network_t *nw, Query *_Q);

        ~Coordinator() override;

        network_t *Net();

        const ProtocolConfig &Cfg() const;

        /** Initialize learner and  variables */
        void InitializeGlobalLearner();

        void SetupConnections();

        /** Initialize a new round */
        void StartRound();

        void FinishRound();

        void FinishRounds();

        /** Rebalance algorithm by Kamp */
        void KampRebalance(node_t *n);

        /** Printing and saving the accuracy */
        void Progress();

        /** Getting the accuracy of the global learner */
        double Accuracy();

        /** Get the communication statistics of experiment */
        vector<size_t> Statistics() const;

        /** Get a model of a node */
        // TODO: uncomment FetchUpdates
//        void FetchUpdates(node_t *n);

        /** Remote call on host violation */
        oneway LocalViolation(sender<node_t> ctx);
        // TODO: uncomment Drift
//        oneway Drift(sender<node_t> ctx, size_t cols);

    };

    struct CoordinatorProxy : remote_proxy<Coordinator> {
        using coordinator_t = Coordinator;
        REMOTE_METHOD(coordinator_t, LocalViolation);
        // TODO: uncomment Drift
//        REMOTE_METHOD(coordinator_t, Drift);

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
        typedef Query query_t;

        query_t *Q;                         // The query management object
        Safezone szone;                     // The safezone object
        RNNLearner *_learner;               // The learning algorithm

        arma::mat drift;                    // The drift of the node

        int num_sites;                      // Number of sites

        size_t datapoints_seen;             // Number of points the node has seen since the last synchronization
        coord_proxy_t coord;                // The proxy of the coordinator/hub

        /** Constructor */
        LearningNode(network_t *net, source_id hid, query_t *_Q);

        const ProtocolConfig &Cfg() const;

        void InitializeLearner();

        void SetupConnections();
        // TODO: uncomment UpdateDrift
//        void UpdateDrift(vector<arma::mat *> &params);
        // TODO: uncomment UpdateStream
//        void UpdateStream(arma::mat &batch, arma::mat &labels);

        /**
         * Remote Methods
         */

        /** Call at the start of a round */
        oneway Reset(const Safezone &newsz);

        /** Transfer data to the coordinator */
        // TODO: (!)uncomment GetDrift
//        ModelState GetDrift();

        /** Set the drift vector (for rebalancing) */
        void SetDrift(ModelState mdl);
        // TODO: uncomment SetGlobalParameters
//        oneway SetGlobalParameters(const ModelState &SHParams);

    };

    struct LearningNodeProxy : remote_proxy<gm_network::LearningNode> {
        typedef gm_network::LearningNode node_t;
        REMOTE_METHOD(node_t, Reset);
        // TODO: uncomment GetDrift
//        REMOTE_METHOD(node_t, GetDrift);
        REMOTE_METHOD(node_t, SetDrift);
        // TODO: uncomment SetGlobalParameters
//        REMOTE_METHOD(node_t, SetGlobalParameters);

        explicit LearningNodeProxy(process *p) : remote_proxy<node_t>(p) {}
    };


} // gm_protocol

namespace dds {
    template<>
    inline size_t byte_size<gm_network::LearningNode *>(
            gm_network::LearningNode *const &) { return 4; }
}

#endif